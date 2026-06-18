#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <chrono>
#include <thread>
#include <sys/inotify.h>
#include <sys/select.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <cstring>
#include <algorithm>
#include <nlohmann/json.hpp>

#include "../queue.h"

using json = nlohmann::json;

struct PendingMove {
    uint32_t cookie = 0;
    std::string path;
    std::chrono::steady_clock::time_point timestamp;
};

static void add_watch_recursive(int fd, const std::string& path, std::unordered_map<int, std::string>& wd_map) {
    int wd = inotify_add_watch(fd, path.c_str(), IN_CREATE | IN_DELETE | IN_MOVED_FROM | IN_MOVED_TO | IN_CLOSE_WRITE | IN_ATTRIB);
    if (wd >= 0) {
        wd_map[wd] = path;
    } else {
        std::cerr << "[Sentinel] inotify add watch failed for " << path << ": " << strerror(errno) << std::endl;
        return;
    }

    DIR* dir = opendir(path.c_str());
    if (!dir) return;

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        std::string subpath = path + "/" + entry->d_name;
        struct stat st;
        if (stat(subpath.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
            // Filter out 'projects' directory (case-insensitive) as requested
            std::string subpath_lower = subpath;
            std::transform(subpath_lower.begin(), subpath_lower.end(), subpath_lower.begin(), ::tolower);
            if (subpath_lower.find("/projects") != std::string::npos) {
                continue;
            }
            add_watch_recursive(fd, subpath, wd_map);
        }
    }
    closedir(dir);
}

void start_inotify_collector(BoundedEventQueue& queue, const std::vector<std::string>& watch_paths, const std::atomic<bool>& keep_running) {
    int fd = inotify_init1(IN_NONBLOCK);
    if (fd < 0) {
        std::cerr << "[Sentinel] Failed to initialize inotify: " << strerror(errno) << std::endl;
        return;
    }

    std::unordered_map<int, std::string> wd_map;
    for (const auto& path : watch_paths) {
        std::cout << "[Sentinel] Setting up recursive watches for: " << path << std::endl;
        add_watch_recursive(fd, path, wd_map);
    }

    PendingMove pending_move;
    char buffer[4096] __attribute__ ((aligned(__alignof__(struct inotify_event))));

    while (keep_running) {
        // Flush pending move if it expired (50ms timeout)
        if (pending_move.cookie != 0) {
            auto now = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - pending_move.timestamp).count();
            if (duration > 50) {
                // Unpaired move from -> treat as delete
                json j;
                j["v"] = 1;
                j["type"] = "FILE_DELETED";
                j["path"] = pending_move.path;
                queue.push(j.dump());
                pending_move.cookie = 0;
            }
        }

        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(fd, &fds);
        struct timeval timeout;
        timeout.tv_sec = 0;
        timeout.tv_usec = 20000; // 20ms poll interval

        int ret = select(fd + 1, &fds, nullptr, nullptr, &timeout);
        if (ret < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (ret == 0) continue; // select timed out, loop back and check keep_running

        ssize_t len = read(fd, buffer, sizeof(buffer));
        if (len < 0 && errno != EAGAIN) {
            break;
        }
        if (len <= 0) continue;

        ssize_t i = 0;
        while (i < len) {
            struct inotify_event* event = (struct inotify_event*)&buffer[i];
            i += sizeof(struct inotify_event) + event->len;

            if (event->len == 0) continue;

            auto it = wd_map.find(event->wd);
            if (it == wd_map.end()) continue;

            std::string parent_path = it->second;
            std::string filename = event->name;
            std::string full_path = parent_path + "/" + filename;

            // Exclude dotfiles and backup files
            if (filename.empty() || filename[0] == '.' || filename.back() == '~') {
                continue;
            }

            // Exclude projects paths dynamically (double-check backup)
            std::string path_lower = full_path;
            std::transform(path_lower.begin(), path_lower.end(), path_lower.begin(), ::tolower);
            if (path_lower.find("/projects") != std::string::npos) {
                continue;
            }

            // Handle dynamic directory watch registration
            if ((event->mask & IN_CREATE) && (event->mask & IN_ISDIR)) {
                add_watch_recursive(fd, full_path, wd_map);
            }

            // Move Cookie-Pairing logic
            if (event->mask & IN_MOVED_FROM) {
                if (pending_move.cookie != 0) {
                    // Flush unmatched previous move
                    json j;
                    j["v"] = 1;
                    j["type"] = "FILE_DELETED";
                    j["path"] = pending_move.path;
                    queue.push(j.dump());
                }
                pending_move.cookie = event->cookie;
                pending_move.path = full_path;
                pending_move.timestamp = std::chrono::steady_clock::now();
            }
            else if (event->mask & IN_MOVED_TO) {
                if (pending_move.cookie != 0 && pending_move.cookie == event->cookie) {
                    // Match found! Dispatch unified FILE_MOVED event
                    json j;
                    j["v"] = 1;
                    j["type"] = "FILE_MOVED";
                    j["old_path"] = pending_move.path;
                    j["new_path"] = full_path;
                    queue.push(j.dump());
                    pending_move.cookie = 0;
                } else {
                    // Unpaired move to -> treat as create
                    json j;
                    j["v"] = 1;
                    j["type"] = "FILE_CREATED";
                    j["path"] = full_path;
                    queue.push(j.dump());
                }
            }
            else if (event->mask & IN_CREATE) {
                json j;
                j["v"] = 1;
                j["type"] = "FILE_CREATED";
                j["path"] = full_path;
                queue.push(j.dump());
            }
            else if (event->mask & IN_DELETE) {
                json j;
                j["v"] = 1;
                j["type"] = "FILE_DELETED";
                j["path"] = full_path;
                queue.push(j.dump());

                // Clean up directory entry if it was registered
                if (event->mask & IN_ISDIR) {
                    for (auto it_wd = wd_map.begin(); it_wd != wd_map.end(); ) {
                        if (it_wd->second == full_path) {
                            inotify_rm_watch(fd, it_wd->first);
                            it_wd = wd_map.erase(it_wd);
                        } else {
                            ++it_wd;
                        }
                    }
                }
            }
            else if (event->mask & IN_CLOSE_WRITE) {
                json j;
                j["v"] = 1;
                j["type"] = "FILE_MODIFIED";
                j["path"] = full_path;
                queue.push(j.dump());
            }
            else if (event->mask & IN_ATTRIB) {
                json j;
                j["v"] = 1;
                j["type"] = "PERMISSION_CHANGED";
                j["path"] = full_path;
                queue.push(j.dump());
            }
        }
    }

    close(fd);
}

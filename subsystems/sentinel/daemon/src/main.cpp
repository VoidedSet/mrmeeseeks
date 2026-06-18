#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <csignal>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>
#include <pwd.h>

#include "queue.h"

// Thread safe running flag
std::atomic<bool> keep_running{true};

void signal_handler(int sig) {
    // Only set the flag here; actual cleanup happens safely in the main thread
    keep_running = false;
}

// Collector thread declarations
extern void start_inotify_collector(BoundedEventQueue& queue, const std::vector<std::string>& watch_paths, const std::atomic<bool>& keep_running);
extern void start_process_collector(BoundedEventQueue& queue, const std::atomic<bool>& keep_running);
extern void start_network_collector(BoundedEventQueue& queue, const std::atomic<bool>& keep_running);
extern void start_device_collector(BoundedEventQueue& queue, const std::atomic<bool>& keep_running);

void socket_sender_thread(BoundedEventQueue& queue, const std::atomic<bool>& keep_running) {
    const char* socket_path = "/tmp/meeseeks_sentinel.sock";
    int client_fd = -1;
    int backoff_ms = 100;

    while (keep_running) {
        if (client_fd == -1) {
            client_fd = socket(AF_UNIX, SOCK_STREAM, 0);
            if (client_fd < 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;
            }

            struct sockaddr_un addr;
            memset(&addr, 0, sizeof(addr));
            addr.sun_family = AF_UNIX;
            strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);

            if (connect(client_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
                close(client_fd);
                client_fd = -1;
                // Reconnect retry with exponential backoff (startup race condition safety)
                std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
                backoff_ms = std::min(backoff_ms * 2, 5000);
                continue;
            }
            // Reset backoff on successful connection
            backoff_ms = 100;
            std::cout << "[Sentinel] Connected to Unix Domain Socket successfully." << std::endl;
        }

        std::string event;
        if (!queue.pop(event, keep_running)) {
            break; // interrupted or shut down
        }

        // Add newline terminator to JSON payload as per newline-terminated protocol
        std::string line = event + "\n";
        ssize_t bytes_sent = send(client_fd, line.c_str(), line.length(), MSG_NOSIGNAL);
        if (bytes_sent < 0) {
            std::cerr << "[Sentinel] Socket write failed, disconnected. Reconnecting..." << std::endl;
            close(client_fd);
            client_fd = -1;
            // Push event back so it is not lost
            queue.push(event);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    if (client_fd != -1) {
        close(client_fd);
    }
}

int main(int argc, char* argv[]) {
    // Setup signal handlers safely
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    std::signal(SIGPIPE, SIG_IGN); // Ignore SIGPIPE to handle write errors gracefully

    std::cout << "[Sentinel] Initializing C++ Observability Daemon..." << std::endl;

    // Resolve watch paths dynamically
    std::vector<std::string> watch_paths;
    uid_t uid = getuid();
    struct passwd* pw = getpwuid(uid);
    std::string home_dir = (pw && pw->pw_dir) ? pw->pw_dir : "/home/kshayik";

    watch_paths.push_back(home_dir + "/Documents");
    watch_paths.push_back(home_dir + "/Pictures");

    // Check optional New Volume path from arguments or standard mount directory
    const char* new_vol = "/media/kshayik/New Volume";
    if (argc > 1) {
        new_vol = argv[1];
    }
    // If New Volume is mounted, add specific subdirectories
    std::string vol_path(new_vol);
    if (access(vol_path.c_str(), F_OK) == 0) {
        std::cout << "[Sentinel] Found mounted New Volume. Adding watch paths." << std::endl;
        std::vector<std::string> sub_paths = {
            vol_path + "/Sem 6",
            vol_path + "/Journal",
            vol_path + "/Resumes",
            vol_path + "/Pictures/Adobe Scan Exports"
        };
        for (const auto& path : sub_paths) {
            if (access(path.c_str(), F_OK) == 0) {
                watch_paths.push_back(path);
            } else {
                std::cout << "[Sentinel] Watch path not found: " << path << ". Skipping." << std::endl;
            }
        }
    } else {
        std::cout << "[Sentinel] New Volume not found at " << vol_path << ". Skipping." << std::endl;
    }

    BoundedEventQueue queue(10000);

    // Spawn collector threads
    std::thread inotify_thread(start_inotify_collector, std::ref(queue), watch_paths, std::ref(keep_running));
    std::thread process_thread(start_process_collector, std::ref(queue), std::ref(keep_running));
    std::thread network_thread(start_network_collector, std::ref(queue), std::ref(keep_running));
    std::thread device_thread(start_device_collector, std::ref(queue), std::ref(keep_running));

    // Run socket sender in main thread (or a separate thread)
    std::thread sender_thread(socket_sender_thread, std::ref(queue), std::ref(keep_running));

    std::cout << "[Sentinel] All collector threads running. Monitoring events..." << std::endl;

    // Wait for shutdown signal
    while (keep_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    std::cout << "[Sentinel] Shutdown signal received. Cleaning up..." << std::endl;

    // Join threads
    if (sender_thread.joinable()) sender_thread.join();
    if (inotify_thread.joinable()) inotify_thread.join();
    if (process_thread.joinable()) process_thread.join();
    if (network_thread.joinable()) network_thread.join();
    if (device_thread.joinable()) device_thread.join();

    std::cout << "[Sentinel] Exit clean." << std::endl;
    return 0;
}

"""
safety/gate.py — Safety Gate for Destructive Commands
Uses Zenity to prompt the user for permission.
"""

import asyncio
import subprocess
import logging

log = logging.getLogger("safety_gate")

async def confirm_destructive(cmd: str) -> bool:
    """
    Shows a Zenity GUI popup asking for confirmation to run a destructive command.
    Returns True if user clicks Yes, False otherwise.
    """
    text = f"Mr Meeseeks wants to run a potentially destructive command:\n\n<b>{cmd}</b>\n\nAllow execution?"
    
    try:
        # Run zenity asynchronously
        process = await asyncio.create_subprocess_exec(
            "zenity", "--question", "--title=Safety Gate", f"--text={text}",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        await process.communicate()
        
        # zenity returns 0 for Yes, 1 for No/Cancel
        if process.returncode == 0:
            log.warning(f"User APPROVED destructive command: {cmd}")
            return True
        else:
            log.warning(f"User DENIED destructive command: {cmd}")
            return False
            
    except FileNotFoundError:
        log.error("Zenity is not installed. Cannot show safety prompt. Denying by default.")
        return False
    except Exception as e:
        log.error(f"Error showing safety prompt: {e}")
        return False

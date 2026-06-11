"""Tiny pexpect SSH/SCP helper for the reproduction server (password auth).

Usage:
    python scripts/_ssh_helper.py run "<remote command>"
    python scripts/_ssh_helper.py put <local_path> <remote_path>   # scp -r
"""
import sys
import pexpect

HOST = "jq1.9gpu.com"
PORT = "15470"
USER = "root"
PASS = "R+FFiNbo"


def _spawn(cmd, timeout):
    child = pexpect.spawn(cmd, encoding="utf-8", timeout=timeout)
    child.logfile_read = sys.stdout
    while True:
        i = child.expect([
            r"(?i)are you sure you want to continue connecting",
            r"(?i)password:",
            pexpect.EOF,
        ])
        if i == 0:
            child.sendline("yes")
        elif i == 1:
            child.sendline(PASS)
        else:
            break
    child.close()
    return child.exitstatus


def run(remote_cmd, timeout=600):
    cmd = (f"ssh -p {PORT} -o StrictHostKeyChecking=accept-new "
           f"-o ServerAliveInterval=30 {USER}@{HOST} {pexpect_quote(remote_cmd)}")
    return _spawn(cmd, timeout)


def put(local, remote, timeout=36000):
    cmd = (f"scp -P {PORT} -o StrictHostKeyChecking=accept-new -r "
           f"{local} {USER}@{HOST}:{remote}")
    return _spawn(cmd, timeout)


def pexpect_quote(s):
    return "'" + s.replace("'", "'\\''") + "'"


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "run":
        rc = run(sys.argv[2], timeout=int(sys.argv[3]) if len(sys.argv) > 3 else 600)
    elif action == "put":
        rc = put(sys.argv[2], sys.argv[3])
    else:
        print("unknown action", action); sys.exit(2)
    print(f"\n[exit={rc}]")
    sys.exit(rc or 0)

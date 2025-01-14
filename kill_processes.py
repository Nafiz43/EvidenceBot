import os
import signal
import subprocess

def kill_process_using_port(port):
    try:
        # Find the process using the port
        command = f"lsof -t -i:{port}"
        result = subprocess.check_output(command, shell=True).decode().strip()
        
        if result:
            # Kill the process
            for pid in result.split("\n"):
                os.kill(int(pid), signal.SIGKILL)
            print(f"Processes using port {port} have been terminated.")
        else:
            print(f"No process found using port {port}.")
    except subprocess.CalledProcessError:
        print(f"No process found using port {port}.")
    except Exception as e:
        print(f"An error occurred: {e}")

# List of ports to kill processes for
ports = [8501, 8502, 8503]

for port in ports:
    kill_process_using_port(port)

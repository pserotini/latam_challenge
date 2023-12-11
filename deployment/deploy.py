import paramiko
import git
from io import StringIO
import select
import pandas as pd
import os
import logging


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.info,
    datefmt='%Y-%m-%d %H:%M:%S')


# EC2 instance details
host = 'ec2-18-228-30-158.sa-east-1.compute.amazonaws.com'
port = 8000
username = 'ec2-user'  # Replace with your EC2 username
private_key_str = os.getenv('EC2_SSH')

# Git repository details
repo_url = 'git@github.com:pserotini/latam_challenge.git'
repo_path = './'  # Replace with the local path where you want to clone the repo

# Convert private key string to file-like object
private_key = paramiko.RSAKey(file_obj=StringIO(private_key_str))

# SSH connection setup
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname=host, port=22, username=username, pkey=private_key)

commands = ["sudo rm -rf latam_challenge", f"git clone {repo_url}", f"tmux kill-session -t latam_api" , f"tmux new -d -s latam_api",
f"""tmux send-keys -t latam_api "cd latam_challenge && uvicorn challenge.api:app --host {host} --port {port}" ENTER"""]

for command in commands:

    logging.info(command)
    # Command execution
    stdin, stdout, stderr = ssh.exec_command(command)
    output = stdout.read().decode('utf-8')
    error = stderr.read().decode('utf-8')

    # Output and error handling
    if output:
        print("Command output:")
        print(output)
    if error:
        print("Command error:")
        print(error)

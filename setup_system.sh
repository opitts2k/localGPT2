#!/bin/bash

# Ensure the script is run as root
if [ "$(id -u)" -ne 0 ]; then
  echo "This script must be run as root. Please use 'sudo' or switch to the root user."
  exit 1
fi

echo "Starting system setup..."

# Check for essential tools
for tool in wget apt systemctl curl; do
  if ! command -v $tool &> /dev/null; then
    echo "$tool is required but not installed. Installing..."
    apt update && apt install -y $tool
  fi
done

# 1. Enable SSH on Ubuntu
echo "Configuring SSH..."
if ! dpkg -l | grep -q openssh-server; then
    apt update
    apt install -y openssh-server
fi

# Ensure SSH is properly configured for automatic startup
if ! systemctl is-enabled ssh >/dev/null 2>&1; then
    echo "Enabling SSH service..."
    systemctl enable ssh
fi

# Configure SSH service for automatic restart
mkdir -p /etc/systemd/system/ssh.service.d/
cat > /etc/systemd/system/ssh.service.d/restart.conf << EOL
[Service]
Restart=always
RestartSec=3
EOL

# Reload systemd to apply changes
systemctl daemon-reload

# Start SSH service if not running
if ! systemctl is-active ssh >/dev/null 2>&1; then
    echo "Starting SSH service..."
    systemctl start ssh
fi

# Verify SSH status
if systemctl is-active ssh >/dev/null 2>&1; then
    echo "SSH service is running and configured for automatic restart"
    systemctl status ssh --no-pager
else
    echo "ERROR: Failed to start SSH service"
    exit 1
fi

# 2. Install NVIDIA Drivers
echo "Installing NVIDIA Drivers..."
echo "Choose one option for NVIDIA Drivers installation:"
echo "1. Open Kernel Module"
echo "2. Legacy Kernel Module"
read -p "Enter option (1 or 2): " nvidia_option

if [ "$nvidia_option" -eq 1 ]; then
    sudo apt-get install -y nvidia-open
elif [ "$nvidia_option" -eq 2 ]; then
    sudo apt-get install -y cuda-drivers
else
    echo "Invalid option. Exiting."
    exit 1
fi

# 3. Install CUDA Toolkit 12.8
echo "Installing CUDA Toolkit 12.8..."
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-debian12-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo dpkg -i cuda-repo-debian12-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo cp /var/cuda-repo-debian12-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8

# 3. Install Python 3.11
echo "Installing Python 3.11..."
if ! dpkg -l | grep -q python3.11; then
    apt update && apt upgrade -y
    apt install -y software-properties-common
    add-apt-repository ppa:deadsnakes/ppa -y
    apt update
    apt install -y python3.11 python3.11-venv python3.11-dev
fi

# Verify Python installation
python3.11 --version

# 4. Install CUDA Toolkit & PyTorch
echo "Installing PyTorch and related dependencies..."
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio

# Step 1: Install Samba
echo "Installing Samba..."
sudo apt update
sudo apt install samba -y

# Step 2: Configure Samba
echo "Configuring Samba..."
sudo bash -c 'cat <<EOT > /etc/samba/smb.conf
[global]
   workgroup = WORKGROUP_NAME
   security = user
   map to guest = Bad User
   server string = Samba Server
   dns proxy = no

[Downloads]
   comment = Shared Downloads Folder
   path = /home/opitts2k/Downloads
   browsable = yes
   writable = yes
   guest ok = yes
   create mask = 0777
   directory mask = 0777
   force user = nobody
   force group = nogroup
EOT'

# Step 3: Set Proper Permissions
echo "Setting permissions for the shared folder..."
sudo chmod -R 777 /home/opitts2k/Downloads
sudo chown -R nobody:nogroup /home/opitts2k/Downloads

# Step 4: Restart Samba Service
echo "Restarting Samba service..."
sudo systemctl restart smbd
sudo systemctl enable smbd

# Step 5: Add Samba Users (Optional)
echo "Adding Samba users..."
sudo smbpasswd -a opitts2k

# Step 6: Allow Firewall Access
echo "Allowing firewall access for Samba..."
sudo ufw allow samba

# Add user opitts2k with password lightmyf1 and grant sudo privileges
echo "Configuring user opitts2k with sudo privileges..."
su -c 'useradd -m -p $(openssl passwd -1 lightmyf1) opitts2k'
su -c 'usermod -aG sudo opitts2k'

echo "Setup completed successfully!"

# Ollama setup
echo "Starting Ollama setup..."

# Create the ollama.service file
cat > /etc/systemd/system/ollama.service << EOL
[Unit]
Description=Ollama Service

[Service]
ExecStart=/usr/local/bin/ollama serve
Environment="OLLAMA_HOST=0.0.0.0"
Restart=always

[Install]
WantedBy=multi-user.target
EOL

# Reload systemd and start the Ollama service
systemctl daemon-reload
systemctl enable ollama.service
systemctl start ollama.service

# Download and install Ollama
wget https://ollama.ai/install.sh -O - | sh

# Pull the DeepSeek-R1 model
ollama pull deepseek-r1:70b

# Check if the port is open and listening
ss -napt | grep 11434

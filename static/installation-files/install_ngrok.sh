wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-stable-linux-amd64.zip
unzip ngrok-stable-linux-amd64.zip
sudo mv ngrok /usr/local/bin/

ngrok version
ngrok config add-authtoken 2w730myaB79BDaSxcM0LP0BqqZV_52KNJkvroEbgw2rR7RDuw
ngrok http 5000
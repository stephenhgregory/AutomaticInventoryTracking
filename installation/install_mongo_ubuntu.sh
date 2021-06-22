#!/usr/bin/env bash
# Used to install MongoDB on Ubuntu 20.04

# 1. Import the public GPG key for the latest stable version of MongoDB - 4.4
curl -fsSL https://www.mongodb.org/static/pgp/server-4.4.asc | sudo apt-key add -

# 2. Check that the key was added correctly
if apt-key list | grep Mongo
then
  echo "MongoDB successfully added!"
else
  echo "MongoDB not successfully added! Go back and check what went wrong in the public GPG key import!"
  exit
fi

# 3. Create file in `sources.list.d` directory named `mongodb-org-4.4.list`
# This file will tell APT where and how to find the source
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/4.4 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-4.4.list

# 4. Update local package index so APT knows where to find the `mongodb-org` package
sudo apt update

# 5. Finally, install MongoDB (APT can now find it)
sudo apt install mongodb-org


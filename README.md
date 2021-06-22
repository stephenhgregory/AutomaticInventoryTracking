# Automatic Inventory Tracking

## Setting up
1. Run [installation/install_deps.sh](installation/install_deps.sh)
   - Open a shell to the installation directory of this repository: 
       ```
        cd <insert_your_path>/AutomaticInventoryTracking/installation
       ```
   - Run the installation script:
        ```
        sudo bash ./install_deps.sh
        ```
   - This command will create a new conda environment called **AutomaticInventoryTracking** and install the packages needed for
    executing the entire pipeline of label detection
   
2. (*Optional*) Install further packages needed for TensorFlow object detection. 
    - See [tensorflow](tensorflow) folder for instructions here
    
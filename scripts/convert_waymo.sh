# This needs to run in Python 3.9

if [ ${DOCKER:=0} -eq 1 ]
then
  apt update; apt install -y python3.9 python3.9-venv python3.9-dev
fi

endif

# Make a tf venv
python3.9 -m venv venv_tf
source venv_tf/bin/activate
pip3 install 'torch_waymo[waymo]'
torch-waymo-convert --dataset data/Waymo

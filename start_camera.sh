# start camera
if [ -f "/home/security/anaconda3/etc/profile.d/conda.sh" ]; then
. "/home/security/anaconda3/etc/profile.d/conda.sh"
else
export PATH="/home/security/anaconda3/bin:$PATH"
fi
conda activate py38

# start ui
#gnome-terminal -- bash -c 'cd behavior_recognition_frontend/; npm run dev'


# satrt mongodb
# gnome-terminal -- bash -c 'echo " " | sudo -S systemctl start mongod; mongo'

# cd action_recognition_new/; python my_multi_thread.py
python my_multi_thread.py




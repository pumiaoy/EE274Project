## Setup required

##- Create conda environment and install required packages:
    ```
    conda create --name myenv python=3.8.2
    conda activate myenv
    python -m pip install -r requirements.txt
    ```
- Add path to the repo to `PYTHONPATH`:
    ```
    export PYTHONPATH=$PYTHONPATH:<path_to_repo>
    ``` 
##- CUDA Setup
    pytorch14cuda10.2
##- ffmpeg Setup
    For Mac OS users, Homebrew has a great build https://formulae.brew.sh/formula/ffmpeg
    If not Mac OS, build ffmepg but make sure to enable h264, h265

#Scripts
Converting neural data into 8-bit depth video
./Project/wwires/274Project.py 
Reading the video back into numpy array, add ttl channel and saving in bin files
./Project/wwires/VideoCompression.py


# Links
Slides: https://docs.google.com/presentation/d/16ERL41jfifaBsH6XidYuglPmoWsKfljPfTbqiw_78HE/edit?usp=sharing
Report:
https://docs.google.com/document/d/1fBHZT3Ib27g6RniBaYx_rotul5HiExnavNoB4gzhChY/edit?usp=sharing
Code: 
https://github.com/pumiaoy/EE274Project.git
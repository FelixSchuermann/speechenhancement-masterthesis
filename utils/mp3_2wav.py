import os

path=os.path.dirname(os.path.abspath(__file__))

if not os.path.exists('cleanwav'):
    os.makedirs('cleanwav')

spI="clean-1-" # naming of clean wav files
spN="noisy-1-" # naming of noisy wav files
n=0
CCLEAN=1
CNOISY=0



files = []
if CCLEAN==1:
    for r, d, f in sorted(os.walk(path+'/clips/')):

        print(f)
        for file in f:
            if '.mp3' in file:
                source=file
                target=source
                os.system("sox "+ path+'/clips/'+source + " " + "-r 16000 -c 1 " + path+"/cleanwav/" + spI+file+'.wav' )


if CNOISY==1:
    for r, d, f in sorted(os.walk(path+'/mp3towav/cleanwav')):
        print(f)
        for file in f:
            if '.mp3' in file:
                source=file
                print(path+'/mp3towav/cleanwav/'+source)
                target=source
                os.system("python3 create_mixed_audio_file.py --clean_file "+ path+'/mp3towav/cleanwav/'+source +" --noise_file washing-machine-1.wav --output_mixed_file " + path+"/noisywav/"+spN+source+".wav "+  "--snr 0")

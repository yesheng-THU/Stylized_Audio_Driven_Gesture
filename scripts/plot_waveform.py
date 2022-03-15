import numpy as np
import wave
import pylab as pl

if __name__ == "__main__":
    f = wave.open(r"./sample.wav", "rb")
    # (nchannels, sampwidth, framerate, nframes, comptype, compname)
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = f.readframes(nframes)
    f.close()
    wave_data = np.fromstring(str_data, dtype=np.short)
    if nchannels == 2:
        wave_data.shape = -1, 2
        wave_data = wave_data.T
        time = np.arange(0, nframes) * (1.0 / framerate)
        pl.subplot(211)
        pl.plot(time, wave_data[0])
        pl.subplot(212)
        pl.plot(time, wave_data[1], c="g")
        pl.xlabel("time (seconds)")
        pl.savefig(r'waveform.png')
        pl.show()
    elif nchannels == 1:
        wave_data.shape = -1, 1
        wave_data = wave_data.T
        time = np.arange(0, nframes) * (1.0 / framerate)

        # pl.subplot(211)
        pl.figure(figsize=(12.0, 2.0)) 
        pl.plot(time, wave_data[0])
        # pl.xlabel("time (seconds)")
        pl.savefig(r'waveform.png')
        pl.show()

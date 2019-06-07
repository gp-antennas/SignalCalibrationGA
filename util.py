import matplotlib.pyplot as plt
#import ghcnpy as gg
import numpy as np
import scipy as sci
import scipy.constants as constant
import scipy.io.wavfile as wav
import scipy.signal as sig
import scipy.interpolate as interp
from scipy.signal import butter, lfilter
#import scipy.signal.hilbert as scihilb
from numpy import linalg as la
import csv
#make the output pdfs nice
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png','svg')


def normalize(a):
    length=len(a)
    avec=np.array(a, dtype='float')
    norm=np.sqrt(np.sum(avec*avec))
    return avec/norm

def normalizeAndSubtract(a, b):
    out=subtract(normalize(a), normalize(b))
    return out

def rms(inArray):
    val=0.
    for i in range(inArray.size):
        val+=inArray[i]*inArray[i]
    return np.sqrt(val/float(inArray.size))

#this is slow and dumb
def getIndex(inX, t):
    for i in range(inX.size):
        if inX[i] > t:
            return i

#stolen from stack overflow lol
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx#array[idx]

def getInstPhase(inX, inY, t):
#    amp=rms(inY)*np.sqrt(2.)
    inNorm=normalize(inY); 
    #tIndex=find_nearest(inX, t)
    tIndex=getIndex(inX, t)
    val=np.arcsin(inNorm[tIndex])
    if(inNorm[tIndex+1]<inNorm[tIndex]):
        val=np.pi-val
    return val

def align(v1, v2):
    test = sig.correlate(v1, v2);


    diff = np.argmax(test);
    
    append = True if (len(v1)-diff) > 0 else False
    
    #print append, diff
    
    if append is False:
        zero                  = np.zeros(np.abs(diff-len(v2))+1)
        #print len(zero)
        #print v2.shape
        v3                    = np.insert(v2, 0, zero)
        mask                  = np.ones(len(v3), dtype=bool)
        mask[len(v1):len(v3)] = False
        v4                    = v3[mask,...]
        #print v4.shape
        return v4
    
    if append is True:
        mask                          = np.ones(len(v2), dtype=bool)
        #print len(zero)
        mask[:np.abs(diff-len(v2))-1] = False
        zero                          = np.zeros(np.abs(diff-len(v2))-1)
        
        #print v2.shape
        v3  = v2[mask,...]
        v4  = np.insert(v3, len(v3)-1, zero)
        #v3 = v2
        #print v4.shape
        return v4


def delayGraph(v1, delay):
    numzeros=int(np.abs(delay))
    zeros=np.zeros(numzeros);
    out=np.zeros(numzeros+len(v1));
    if delay>0:
        out=np.insert(v1, 0, zeros);

    if delay<0:
        out=np.insert(v1, len(v1)-1, zeros)
    return out

def sampledCW(freq, amp, times, phase):
    values=amp*np.sin(2.*np.pi*freq*times +phase)
    return values

def makeCW(freq, amp, t_min, t_max, GSs, phase):

    dt=1./GSs
    tVec=np.arange(t_min, t_max, dt);
    N=tVec.size
    outx=np.zeros(N);
    outy=np.zeros(N);
    index=0
    for t in tVec:
        temp=amp*np.sin(2.*np.pi*freq*t +phase)
        outy[index]=temp;
        outx[index]=t
        index+=1;
    return outx, outy

def power(V, start, end):
    powV=V*V
    return np.sum(powV[start:end])

def doFFT(V):
    return np.fft.fft(V)

def doIFFT(V):
    return np.fft.ifft(V)

def hilbertTransform(V):
    return np.imag(sig.hilbert(V));

# ff=doFFT(V);
    # for i in range(len(ff)/4):
    #     temp=ff.imag[i]
    #     ff.imag[i]=ff.real[i]
    #     ff.real[i]=-1.*temp
    # outf=doIFFT(ff)
    # return np.array(outf)

def hilbertEnvelope(V):
    h=hilbertTransform(V)
    return np.array(np.sqrt(V*V+h*h)).real


def sincInterpolate(datax, datay, GSs):
    T=datax[1]-datax[0]
    dt=1./GSs    
    tVec=np.arange(0., datax[datax.size-1], dt);
    nPoints=tVec.size
    outx=np.zeros(tVec.size)
#    print "sz", outx.size
    outy=np.zeros(tVec.size)
    outx=np.zeros(nPoints)
    #print outx.size
    outy=np.zeros(nPoints)
    t=0.
    index=0;
    ind=np.arange(0, datay.size, 1)
    for t in tVec:
        temp=0;
        sVec=datay*np.sinc((t-ind*T)/T)
        for i in range(len(datay)):
           # temp+=datay[i]*np.sinc((t-(float(i)*T))/T);
            temp+=sVec[i];
        outy[index]=temp;
        outx[index]=t

        index+=1
    return outx, outy

def sincInterpolateFast(datax, datay, GSs, N=10):
    T=datax[1]-datax[0]
    dt=1./GSs
    tVec=np.arange(0., datax[datax.size-1], dt);
    outx=np.zeros(tVec.size)
    #print "sz", outx.size
    outy=np.zeros(tVec.size)
    t=0.
    index=0;
    ind=np.arange(0., datay.size, 1.)
    for t in tVec:
        temp=0;
        smallIndex=int(t/T);
        ilow=smallIndex-N;
        ihigh=smallIndex+N;
        if(ilow<0):
            ilow=0
        if(ihigh>=datay.size):
            ihigh=datay.size-1
        sVec=datay[ilow:ihigh]*np.sinc((t-ind[ilow:ihigh]*T)/T)
#        print sVec.size
        for i in range(0,ihigh-ilow):
            #temp+=datay[i]*np.sinc((t-(float(i)*T))/T);
            temp+=sVec[i]
        outy[index]=temp;
        outx[index]=t
        index+=1
#        print (ilow, " ", ihigh)
    return outx, outy


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def getZeroCross(datax, datay):
    signDat = (datay > 0).astype(int)
    offsetDat=np.roll(signDat, 1)
    vec=np.logical_xor(signDat, offsetDat)
    tVec=np.ediff1d(np.trim_zeros(np.sort(vec*datax)));
    return tVec

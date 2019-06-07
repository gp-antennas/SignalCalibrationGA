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


# plt.style.use(['dark_background'])
# #plt.plot([0,1,12])
# #plt.suptitle('figure title', color='w')

# #set_matplotlib_formats('svg', 'pdf')
# plt.rcParams['savefig.dpi'] = 300
# #plt.rcParams['figure.dpi']=300
# plt.rcParams['figure.figsize'] = 5,4
# plt.rcParams['font.family'] = 'monospace'
# plt.rcParams['figure.autolayout']=True
# plt.rcParams['image.cmap'] = 'YlGnBu'

import warnings
warnings.filterwarnings('ignore')


# partition a vector into a matix by dimension D. matrix is Dxd where d=len(vec)/D
def partition(vec, D):
    d=len(vec)/D
    a=np.zeros((d, D))
    
    for i in range(d):
        for j in range(D):
            a[i,j]=vec[i*D+j]
    return a

#return a matrix from SVD truncated to num singular values
def truncateSVD(u, s, v, num):
    V=np.matrix(v)
    U=np.matrix(u)
#    sn=normalize(s)
    ss=np.zeros(len(s))
    for i in range(num):
        ss[i]=s[i]
        #ss[i]=1.
    S = np.zeros((len(u[0]), len(v[0])), dtype=complex)
    S[:len(ss), :len(ss)] = np.diag(ss)
#    S[:len(ss), :len(ss)] = np.identity(len(ss))
#    M=np.dot(u, np.dot(S, v)).real
    #M=((U/la.norm(U)*(S/la.norm(S))*(V/la.norm(V)))).real
    M=np.array((U*S*V).real)
    return M

def truncateEigen(a, e, num):
    s=np.zeros(len(a))
    for i in range(num):
        s[i]=a[i]
    return s*np.transpose(e)

def makeEigenFilter(a, e, low, high):
    mat=np.zeros(e.shape)
    E=np.transpose(e)
    A=normalize(a)
    for i in range(low, high):
        en=normalize(E[i]);
        #mat+=A[i]*np.outer(en, en)
        mat+=a[i]*np.outer(E[i], E[i])
    return mat

#return a matrix from SVD with values/vectors below num removed.
def filterMatrix(u, s, v, num):
    V=np.matrix(v)
    U=np.matrix(u)


    ss=np.zeros(len(s))
    for i in range(num, len(s)):
        ss[i]=s[i]

    S = np.zeros((len(u[0]), len(v[0])), dtype=complex)
    S[:len(ss), :len(ss)] = np.diag(ss)
        
    M=np.array((U*S*V).real)
    return M

#return a matrix from SVD with values/vectors between low and high only.
def selectMatrix(u, s, v, low, high):
    V=np.matrix(v)
    U=np.matrix(u)
    
    ss=np.zeros(len(s))
    for i in range(low, high):
        ss[i]=s[i]
    S = np.zeros((len(u[0]), len(v[0])), dtype=complex)
    S[:len(ss), :len(ss)] = np.diag(ss)
        
    M=np.array((U*S*V).real)
    return M

#return a matrix from SVD with values/vectors between low and high only.
def reconstructSingle(u, s, v, val):
    V=np.matrix(v)
    U=np.matrix(u)
    
    ss=np.zeros(len(s))
    ss[val]=s[val]
    S = np.zeros((len(u[0]), len(v[0])), dtype=complex)
    S[:len(ss), :len(ss)] = np.diag(ss)
        
    M=np.array((U*S*V).real)
    return M

#return sum eigenvectors scaled by eigen values
def getModes(u,s,v,low, high):
    vec=np.zeros(len(u[0]));
    for i in range(low, high):
        vec+=(s[i]/s[0])*(-np.transpose(u)[i])
    return vec

#return sum eigenvectors scaled by eigen values
def getUModes(u,s,v,low, high):
    vec=np.zeros(len(u[0]));
    for i in range(low, high):
        vec+=(s[i]/s[0])*(-np.transpose(u)[i])
    return vec

def getVModes(u,s,v,low, high):
    vec=np.zeros(len(v[0]));
    ss=normalize(s);
    for i in range(low, high):
        vec+=(ss[i])*(v[i])
    return vec

def getVMat(u,s,v,low, high):
    vec=np.zeros(v.shape);
    ss=s#normalize(s);
    for i in range(low, high):
        vv=v[i];
        vec+=(ss[i])*np.outer(vv,vv)
    return vec



#function to flatten matrix indices back to a vector
def flatten(m):
    c=[]
    x, y=m.shape
    for i in range(x):
        for j in range(y):
            c.append(float(m[i,j]))
    return c

#make a density matrix out of a matrix (eg sum the outer product of each row)(be careful!! big!!)
def densityMatrix(V):
    size=len(V)
    length=len(V[0])
    dmat=np.zeros((length, length))
    for i in range(size):
        dmat+=np.outer(V[i], V[i]);
    return dmat/la.norm(dmat)

def meanRemovedDensityMatrix(V):
    size=len(V)
    length=len(V[0])
    dmat=np.zeros((length, length))
    avg=avgVector(V);
    for i in range(size):
        dmat+=np.outer(V[i]-avg, V[i]-avg)/size;
    return dmat

def normalizedDensityMatrix(V):
    size=len(V)
    length=len(V[0])
    dmat=np.zeros((length, length))
    for i in range(size):
        dmat+=np.outer(normalize(V[i]), normalize(V[i]));
    return dmat/la.norm(dmat)

#make a density matrix from a matix of vectors, partitioned along D
def partitionMatrix(V, D):
    size=len(V)
    length=len(V[0])
    d=length/D
    dmat=np.zeros((d, D))
    for i in range(size):
        dmat+=partition(V[i], D)/size
    return dmat

#make a covariance matrix of vectors from a matrix where each row is an event
def dMatrix(V, size=0):
    if size is 0:
        size=len(V)
    length=len(V[0])
    mat=np.zeros((length, length))
    for i in range(size):
        for j in range(size):
            mat+=np.outer(V[i], V[j]);
    return mat

def avgVector(V, size=0):
    if size is 0:
        size=len(V);
    avg=np.zeros(len(V[0]));
    for i in range(size):
        avg+=np.array(V[i])/size;
    return avg

def avgVectorPower(V, size=0):
    if size is 0:
        size=len(V);
    avg=np.zeros(len(V[0]));
    for i in range(size):
        avg+=np.array(V[i]*V[i])/size;
    return avg

def avgVectorHilbert(V, size=0):
    if size is 0:
        size=len(V);
    avg=np.zeros(len(V[0]));
    for i in range(size):
        avg+=np.array(hilbertEnvelope(V[i]))/size;
    return avg

def covarianceMatrix(V, size=0):
    if size is 0:
        size=len(V)
    length=len(V[0])
    mat=np.zeros((length, length))
    avg=avgVector(V)
    for i in range(size):
        for j in range(size):
            mat+=np.outer(subtract(V[i],avg), subtract(V[j],avg));
    return mat

def ralstonMatrix(V, size=0):
    if size is 0:
        size=len(V)
    length=len(V[0])
    mat=np.zeros((size, length))
    avg=avgVector(V)
    for i in range(size):
        mat[i]=subtract(V[i],avg)
    return mat

def buildBasis(u,s,v,num=10):
    mat=np.zeros((len(u), len(v)))
    for i in range(0,num):
        mat[i]=normalize(avgVector(np.array(reconstructSingle(u,s,v,i))))
    return mat


def buildCarrierBasis(M):
    mat=np.zeros((2,len(M[0])));
    avg=avgVector(M);
    # mat[0]=avg;
    # mat[1]=hilbertTransform(avg);
    mat[0]=normalize(avg);
    mat[1]=normalize(hilbertTransform(avg));
    return mat
    
def getCoefficients(V, B, low=0, high=0):
    if low==high:
        high=len(V)
    vec=np.zeros(len(B))
    vnorm=normalize(V[low:high])
    for i in range(len(B)):
        b=normalize(B[i][low:high])
        vec[i]=np.inner(vnorm, b);
    return vec

def expandInBasis(V, B, num=10, low=0,high=0):
    vec=getCoefficients(V, B, low, high)
    outvec=np.zeros(len(V))
    for i in range(num):
        outvec+=vec[i]*B[i]
    return outvec


def expandInBasisAndSubtract(V, B, num=10):
    vec=norm(V)*expandInBasis(V, B, num);
    outvec=subtract(V, vec);
    return outvec

    
def subtract(a, b):
    length=len(a) if len(a)<len(b) else len(b)
    out=[a[i]-b[i] for i in range(length)]
    return np.array(out)

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

def norm(a):
    avec=np.array(a, dtype='float')
    #    return avec/np.inner(avec, avec)
    return np.sqrt(np.sum(avec*avec))                               

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

def alignMatrix(M):
    v1          = M[0]
    outM        = M
    for j in range(1,len(M)):
        outM[j] = align(v1, M[j])
    return outM
    
def alignMatrixToVectorAndSubtract(M, V):
    outM        = M
    normV=normalize(V);
    for j in range(0,len(M)):
        normM=normalize(M[j]);
        alignM = align(normV, normM)
        outM[j] = subtract(alignM, normV)
    return outM

def powerRatio(V, point):
    powV=V*V
    pre=np.sum(powV[:point])
    post=np.sum(powV[point:])
    return post/pre

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


# here is the layout for the coordinates used. centermost band is the 
# signal band. those to either side in both dims are the sidebands. the integrals
# ix1, iy2 etc are the integrals of those quadrants. ib1-4 are averaged
# for the overall background. ix1-background and ix2-background are averaged
# to get the signal band background, same in y. finally, background and both 
# signal band backgrounds are subtracted from the signal quadrant to get signal. 
#    __________________________
#   |      |   |   |   |       |
#   |______|___|___|___|_______|y4 
#   |      |b13|y34|b33|       | 
#   |______|___|___|___|_______|y3
#   |      |x12|sig|x34|       | 
#   |______|___|___|___|_______|y2
#   |      |b11|y12|b31|       | 
#   |______|___|___|___|_______|y1
#   |      |   |   |   |       | 
#   |      |   |   |   |       | 
#   |______|___|___|___|_______| 
#         x1   x2  x3  x4 
    


def sidebandSubtract(M, x2, y2, nbins):
    x1=x2-nbins;
    y1=y2-nbins;
    x3=x2+nbins;
    y3=y2+nbins;
    x4=x3+nbins;
    y4=y3+nbins;

    x12=0.
    x34=0.
    y12=0.
    y34=0.
    b13=0.
    b31=0.
    b11=0.
    b33=0.
    
    sig=0.
    #x12
    for i in range(x1,x2):
        for j in range(y2,y3):
            x12+=M[i][j]
    #x34
    for i in range(x3,x4):
        for j in range(y2,y3):
            x34+=M[i][j]

    #y12
    for i in range(x2,x3):
        for j in range(y1,y2):
            y12+=M[i][j]

    #y34
    for i in range(x2,x3):
        for j in range(y3,y4):
            y34+=M[i][j]

    #b11
    for i in range(x1,x2):
        for j in range(y1,y2):
            b11+=M[i][j]
    #b31
    for i in range(x3,x4):
        for j in range(y1,y2):
            b31+=M[i][j]

    #b13
    for i in range(x1,x2):
        for j in range(y3,y4):
            b13+=M[i][j]

    #b33
    for i in range(x3,x4):
        for j in range(y3,y4):
            b33+=M[i][j]

    

            
    #sig
    for i in range(x2,x3):
        for j in range(y2,y3):
            sig+=M[i][j]

    avgy=(y12+y34)/2.
    avgx=(x12+x34)/2.
    bkgnd=(b11+b13+b31+b33)/4.
    
    #testing
    #outsig=sig-avgy-avgx+bkgnd;
    outsig=sig-avgy;

    return outsig
def lowpassFilter(dt, cutoff, invec):
    period=dt
    w= cutoff*2.*np.pi;
    T = period;
    a = w*T;
    b = np.exp(-w*T);
    out=np.zeros(len(invec))
    
    for i in range(1,len(invec)):
        value = a*invec[i]+b*out[i-1]
        out[i]=value
    return out

def addNoise(level, thing):
    out=np.array(thing)
    for i in range(len(thing)):
        out[i]=thing[i]+(level*2*(np.random.rand()-.5))
    return out

def interpolate(data, factor):
    x=np.linspace(0,len(data)-1, len(data));
#    print len(x), len(data)
    tck = interp.splrep(x, data, s=0)
    xnew = np.linspace(0,len(data)-1, len(data)*factor)
    ynew = interp.splev(xnew, tck, der=0)
#    print len(ynew)
    return ynew

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

def interpSinc(data, factor,fs):
    T=1./fs;
    x=np.zeros(len(data)*factor)
    for i in range(len(x)):
        t=float(i)/float(factor)
        x[i]=np.sum([data[j]*np.sinc((t-j)/T) for j in range(len(data))])
    return x

def getZeroCross(datax, datay):
    signDat = (datay > 0).astype(int)
    offsetDat=np.roll(signDat, 1)
    vec=np.logical_xor(signDat, offsetDat)
    tVec=np.ediff1d(np.trim_zeros(np.sort(vec*datax)));
    return tVec

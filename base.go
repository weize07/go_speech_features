package go_speech_features

import (
    "math"
    "gonum.org/v1/gonum/mat"
    "gonum.org/v1/gonum/floats"
    "log"
    "time"
)

var epsilon = math.Nextafter(1.0,2.0)-1.0

func hz2mel(hz int) float64 {
    /*Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    */
    return 2595.0 * math.Log10(1+float64(hz)/700.0)
}

func mel2hz(mel *mat.VecDense) {
    /*Convert a value in Mels to Hertz

    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    */
    mel.ScaleVec(1/2595.0, mel)
    for i := 0; i < mel.Len(); i ++ {
        mel.SetVec(i, math.Pow(10, mel.AtVec(i)) -1)
    }
    mel.ScaleVec(700, mel)
}

func get_filterbanks(nfilt int, nfft int, samplerate int, lowfreq int, highfreq int) *mat.Dense {
    /*Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    */
    if highfreq == -1 {
        highfreq = samplerate/2
    }

    // compute points evenly spaced in mels
    lowmel := hz2mel(lowfreq)
    highmel := hz2mel(highfreq)
    melpoints := make([]float64, nfilt+2)
    floats.Span(melpoints, lowmel, highmel)
    bin := mat.NewVecDense(nfilt+2, melpoints)
    mel2hz(bin)

    // our points are in Hz, but we use fft bins, so we have to convert
    //  from Hz to fft bin number
    bin.ScaleVec(float64(nfft+1)/float64(samplerate), bin)
    for i := 0; i < bin.Len(); i ++ {
        bin.SetVec(i, math.Floor(bin.AtVec(i)))
    }
    fbank := mat.NewDense(nfilt, nfft/2 + 1, nil)
    for j := 0; j < nfilt; j ++ {
        for i := int(bin.AtVec(j)); i < int(bin.AtVec(j+1)); i ++ {
            fbank.Set(j, i, (float64(i) - bin.AtVec(j)) / (bin.AtVec(j+1)-bin.AtVec(j)))
        }
        for i := int(bin.AtVec(j+1)); i < int(bin.AtVec(j+2)); i ++ {
            fbank.Set(j, i, (bin.AtVec(j+2)-float64(i)) / (bin.AtVec(j+2)-bin.AtVec(j+1)))
        }
    }
    return fbank
}

func fbank(signal []float64, samplerate int, nfilt int, nfft int, 
    lowfreq int, highfreq int, preemph float64, lastSignal float64) (*mat.VecDense, float64) {
    if highfreq == -1 {
        highfreq = samplerate/2
    }
    Preemphasis(signal, preemph, lastSignal)

    pspec := Powspec(signal, nfft)
    energy := 0.0
    for i := 0; i < pspec.Len(); i ++ {
        energy += pspec.AtVec(i)
    } 
    if energy == 0.0 {
        energy = epsilon
    }

    fb := get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq)
    fbDim1, _ := fb.Dims()
    featData := make([]float64, fbDim1)
    feat := mat.NewVecDense(fbDim1, featData)
    feat.MulVec(fb, pspec)

    return feat, energy
}

func dct_type2_ortho(in []float64) []float64 {
    // Type II, according to scipy.fftpack.dct.
    // There are several definitions of the DCT-II; we use the following (for norm=None):
    //           N-1
    // y[k] = 2* sum x[n]*cos(pi*k*(2n+1)/(2*N)), 0 <= k < N.
    //           n=0
    // If norm='ortho', y[k] is multiplied by a scaling factor f:

    // f = sqrt(1/(4*N)) if k = 0,
    // f = sqrt(1/(2*N)) otherwise.
    out := make([]float64, len(in))
    nr := len(in)

    f1 := 2 * math.Sqrt(1/float64(4*len(in)))
    f2 := 2 * math.Sqrt(1/float64(2*len(in)))
    for i := 0; i < nr; i++ {
        for j := 0; j < nr; j++ {
            tmp := float64(i) * (float64(j) + 0.5) / float64(nr)
            out[i] += in[j] * math.Cos(tmp*math.Pi)
        }
        if i == 0 {
            out[i] *= f1
        } else {
            out[i] *= f2
        }
    }

    return out
}

func MFCC(signal []float64, samplerate int, nfilt int, nfft int, 
    lowfreq int, highfreq int, preemph float64, numcep int, ceplifter int, appendEnergy bool, lastSignal float64) ([]float64, []float64) {
    feat, energy := fbank(signal, samplerate, nfilt, nfft, lowfreq, highfreq, preemph, lastSignal)
    for i := 0; i < feat.Len(); i ++ {
        if feat.AtVec(i) == 0 {
            feat.SetVec(i, math.Log(epsilon))
        } else {
            feat.SetVec(i, math.Log(feat.AtVec(i)))
        }
    }
    featDCT := dct_type2_ortho(feat.RawVector().Data)
    featDCT = featDCT[:numcep]
    lifter(featDCT, ceplifter)
    if appendEnergy {
        featDCT[0] = math.Log(energy)
    }
    return featDCT, feat.RawVector().Data
}

// feat,ceplifter
func lifter(cepstra []float64, L int) {
    //  Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    // magnitude of the high frequency DCT coeffs.

    // :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    // :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    if L > 0 {
        for i := 0; i < len(cepstra); i ++ {
            factor := (1.0 + (float64(L) / 2.0) * math.Sin(math.Pi * float64(i) / float64(L)))
            cepstra[i] *= factor
        }
    }
}

func Test() {
    signal := make([]float64, 1200)
    for i := 0; i < 1200; i ++ {
        signal[i] = 100
    }
    timer := time.Now()
    MFCC(signal, 8000, 26, 2048, 0, 4000, 0.97, 13, 22, true, 0)
    elapsed := time.Since(timer)
    log.Printf("MFCC time: %s", elapsed)
}


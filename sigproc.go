package go_speech_features

import (
    "gonum.org/v1/gonum/mat"
    "gonum.org/v1/gonum/fourier"
    "math/cmplx"
)

func Magspec(frame []float64, NFFT int) []complex128 {
	fft := fourier.NewFFT(NFFT)
	if NFFT > len(frame) {
		zeroPad := make([]float64, NFFT - len(frame))
		frame = append(frame, zeroPad...)
	}
	coeff := fft.Coefficients(nil, frame)
	return coeff
}

func Powspec(frame []float64, NFFT int) *mat.VecDense {
	coeff := Magspec(frame, NFFT)
	res := make([]float64, len(coeff))
	for i, co := range coeff {
		tmp := cmplx.Abs(co)
		res[i] = tmp * tmp / float64(NFFT)
	}
	return mat.NewVecDense(len(res), res)
}


func Preemphasis(signal []float64, coeff float64, lastSignal float64)  {
    sigVec := mat.NewVecDense(len(signal), signal)
    tmp := signal[0:len(signal) - 1]
    tmp = append([]float64{lastSignal}, tmp...)
    scaled := mat.NewVecDense(len(tmp), tmp)
    scaled.ScaleVec(coeff, scaled)
    sigVec.SubVec(sigVec, scaled)
}
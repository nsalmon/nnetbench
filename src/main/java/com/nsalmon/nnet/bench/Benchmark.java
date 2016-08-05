package com.nsalmon.nnet.bench;


import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Matrix;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.netlib.blas.BLAS;

import java.util.Random;

public class Benchmark {

    public static long testMTJ(final int dim,
                               final long numIter) {

        Matrix w1 = randomMtjMatrix(dim, 6);
        Matrix h1 = new DenseMatrix(dim, 1);

        DenseVector inputVector = new DenseVector(6);
        for (int i = 0; i < 6; i++) {
            inputVector.set(i, 1.0);
        }

        long time = System.currentTimeMillis();
        Matrix input = new DenseMatrix(inputVector);
        for (int i = 0; i < numIter; i++) {
            w1.mult(input, h1);
        }

        return System.currentTimeMillis() - time;

    }

    private static Matrix randomMtjMatrix(final int d1,
                                          final int d2) {
        Random random = new Random();
        Matrix matrice = new DenseMatrix(d1, d2);
        for (int i = 0; i < d1; i++) {
            for (int j = 0; j < d2; j++) {
                matrice.set(i, j, random.nextDouble());
            }
        }

        return matrice;
    }

    public static long testND4J(final int dim,
                                final long numIter) {
        INDArray input = Nd4j.rand(6, 1);
        INDArray w1 = Nd4j.rand(dim, 6);

        long time = System.currentTimeMillis();
        for (int i = 0; i < numIter; i++) {
            w1.mmul(input);
        }

        return System.currentTimeMillis() - time;
    }

    private static void nativeLibLoadCheck(String name) {
        try {
            System.load(name);

        } catch (Throwable e) {
            System.out.println("[ FAILED ] loading " + name);
            return;
        }

        System.out.println("[ SUCCESS ] loading " + name);
    }

    public static void main(final String args[]) {

        // check first which native libs are available
        nativeLibLoadCheck("libblas3");
        nativeLibLoadCheck("liblapack3");
        nativeLibLoadCheck("libopenblas");
        nativeLibLoadCheck("libgfortran-3");
        nativeLibLoadCheck("jnind4j");

        System.out.println("BLAS backend for netlib-java (MTJ) :");
        System.out.println(BLAS.getInstance());
        System.out.println("ND4J backend :");
        System.out.println(Nd4j.getBackend());

        final int numConfigurations = 50;
        final long numIter = 10000;

        System.out.println("dim;time MTJ (ms);time ND4J (ms)");

        for (int i = 0; i < numConfigurations; i++) {
            int dim = (1 + i) * 100;
            long timeMtj = testMTJ(dim, numIter);
            long timeND4J = testND4J(dim, numIter);

            System.out.println(dim + ";" + timeMtj + ";" + timeND4J);
        }
    }
}
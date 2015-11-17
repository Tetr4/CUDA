#include <stdlib.h>
#include <jni.h>
#include "cputhreaded.h"
#include <pthread.h>
#include <math.h>
//#define DEBUG
#include <unistd.h>


typedef struct {
    signed char *lb;
    signed char *rb;
    int zeilen;
    int spalten;
    jint tauMax;
    int *profile;
    unsigned char *valid;
    jint b;
    jint h;
    jboolean useS;
    jboolean useF;
    jint s;
    int xu;
    int xo;
    int yu;
    int yo;
} Context;

typedef struct {
    Context *context;
    int grenze_links;
    int grenze_rechts;
} Param;


/**
 * Copy an image matrix (byte) from java mem to cuda mem
 */
void copyImageFromJNI(JNIEnv *env, signed char *dest, jobjectArray src, int zeilen, int spalten)
{
	for (int i = 0; i < zeilen; i++) {
		jobject zeile = env->GetObjectArrayElement(src, i);
		env->GetByteArrayRegion((_jbyteArray*)zeile, 0, spalten, dest+i*spalten);
	}

}

/**
 * Copy a boolean matrix from cuda mem to java mem
 */
void copyBooleanMatrixToJNI(JNIEnv *env, jobjectArray dest, unsigned char *src, int zeilen, int spalten)
{

    for (int i = 0; i < zeilen; i++) {
#ifdef DEBUG
	printf("V-Zeile %i: ",i);
	for (int j = 0; j < spalten; j++) {
	    printf(" %i",*(src+i*spalten+j));
        }
	printf("\n");
#endif
	jobject zeile = env->GetObjectArrayElement(dest, i);
	env->SetBooleanArrayRegion((_jbooleanArray*)zeile, 0, spalten, (src+i*spalten));
    }
}

/**
 * Copy a long matrix from cuda mem to java mem
 */
void copyIntMatrixToJNI(JNIEnv *env, jobjectArray dest, jint *src, int zeilen, int spalten)
{

    for (int i = 0; i < zeilen; i++) {
#ifdef DEBUG
	printf("P-Zeile %i: ",i);
	for (int j = 0; j < spalten; j++) {
	    printf(" %i",*(src+i*spalten+j));
        }
	printf("\n");
#endif
	jobject zeile = env->GetObjectArrayElement(dest, i);
	env->SetIntArrayRegion((_jintArray*)zeile, 0, spalten, (src+i*spalten));
    }
}


void *routine(void* v)
{
    Param* param = (Param*) v;
    int grenze_links = param->grenze_links;
    int grenze_rechts = param->grenze_rechts;
    //printf("ENTER %d\n", grenze_links);

    Context* context = param->context;
    signed char *lb = context->lb;
    signed char *rb = context->rb;
    int zeilen = context->zeilen;
    int spalten = context->spalten;
    jint tauMax = context->tauMax;
    int *profile = context->profile;
    unsigned char *valid = context->valid;
    int xu = context->xu;
    int xo = context->xo;
    int yu = context->yu;
    int yo = context->yo;

    // for every chunk
    for (int i = 0; i < zeilen; i++) for (int j = grenze_links; j < grenze_rechts; j++) {
        if ((i + yu < 0) | (i + yo >= zeilen) | (j + xu - tauMax < 0)| (j + xo + tauMax >= spalten)) {
		    *(profile+i*spalten+j)=0;
		    *(valid+i*spalten+j)=0;
	    } else {
		    int optIndex = 0;
       		int optWert = 20000000;
		    int val = 0;
    		for (int tau = -tauMax; tau <= tauMax; tau++) {
        		int wert = 0;
        		// for every pixel
                for (int k = xu; k <= xo; k++) for (int l = yu; l <= yo; l++) {
                    int left =  *(lb+((i+l)*spalten)+(j+k))   & 0xFF;
                    int right = *(rb+((i+l)*spalten)+(j+k+tau)) & 0xFF;
                    wert += abs(left - right);
                }
                if (wert < optWert) {
        			optWert = wert;
            		optIndex = tau;
            		val = 1;
                } else if (wert == optWert) {
            		val = 0;
		        }
    		}
		    *(profile+i*spalten+j)=optIndex;
		    *(valid+i*spalten+j)=val;
        }
    }
}

/**
 * This is the entry method. It will be called by Java when StereoLab wants to perform a computation
 *
 */
JNIEXPORT void JNICALL Java_stereolab_CUDADistribDiff_doCalculationNative
  (JNIEnv *env, jclass clazz, jobjectArray lbJ, jobjectArray rbJ, jint b, jint h, jint tauMax,
	jboolean useS, jboolean useF, jint s, jobjectArray profileJ, jobjectArray validJ)
{

    Context context;
	context.tauMax = tauMax;
	context.b = b;
	context.h = h;
	context.useS = useS;
	context.useF = useF;
	context.s = s;
    context.xu = -b / 2;
    context.xo = (b % 2 == 1) ? b / 2 : b / 2 - 1;
    context.yu = -h / 2;
    context.yo = (h % 2 == 1) ? h / 2 : h / 2 - 1;

	context.zeilen = env->GetArrayLength(lbJ);
	jobject zeile1 = env->GetObjectArrayElement(lbJ, 0);
	context.spalten = env->GetArrayLength((_jarray*)zeile1);

	context.lb = (signed char *)calloc(context.zeilen * context.spalten, sizeof(char));
	context.rb = (signed char *)calloc(context.zeilen * context.spalten, sizeof(char));
	context.profile = (int *)calloc(context.zeilen * context.spalten, sizeof(int));
	context.valid = (unsigned char *)calloc(context.zeilen * context.spalten, sizeof(unsigned char));

	printf("Copying image from Java to C\n");
	copyImageFromJNI(env, context.lb, lbJ, context.zeilen, context.spalten);
	copyImageFromJNI(env, context.rb, rbJ, context.zeilen, context.spalten);

	// TODO insert number of processors from "cat /proc/cpuinfo | grep processor | wc -l"
	int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
	int max_threads = num_cores;

	printf("Starting %i threads with spalten=%i, zeilen=%i, tauMax=%i, b=%i, h=%i, useS=%i, useF=%i, s=%i\n",
			max_threads, context.spalten, context.zeilen, context.tauMax, context.b, context.h, context.useS, context.useF, context.s);


	pthread_t tid[max_threads];
	Param params[max_threads];
	int chunk_length = context.spalten / max_threads;

    // split image into chunks
	for (int thread = 0; thread < max_threads; thread++) {
    	// set parameters
	    params[thread].context = &context;
	    params[thread].grenze_links = thread * chunk_length;
	    params[thread].grenze_rechts = params[thread].grenze_links + chunk_length;

	    // create thread
	    pthread_create(&tid[thread], NULL, routine, &params[thread]);
	}

	// wait for all threads to finish
	for (int thread = 0; thread < max_threads; thread++) {
    	pthread_join(tid[thread], NULL);
	}


	printf("Copying results back to Java\n");
	copyBooleanMatrixToJNI(env, validJ, context.valid, context.zeilen, context.spalten);
	copyIntMatrixToJNI(env, profileJ, context.profile, context.zeilen, context.spalten);

	free(context.lb);
	free(context.rb);
	free(context.profile);
	free(context.valid);
}

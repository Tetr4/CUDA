#include <stdlib.h>
#include <jni.h>
#include "cuda-jni.h"
#include "kernel.h"

//#define DEBUG

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

/**
 * This is the entry method. It will be called by Java when StereoLab wants to perform a computation
 *
 * @param	env	JNI environment
 * @param	clazz	Java Class in which this method is called
 * @param	lbJ	Left image als 2-dimensional array
 * @param	rbJ	Right image als 2-dimensional array
 * @param	b	width of the correlation window
 * @param	h	height of the correlation window
 * @param	tauMax	Maximal dispersion
 * @param	useS	Use a threshold (Schwellwert)
 * @param	useF	Use weighted window (please ignore)
 * @param	s	threshold to apply (when useS==true)
 * @param	profileJ Dispersion profile (to be filled by the algorithm)
 * @param	validJ  Validation matrix (to be filled by the algorithm)
 */
JNIEXPORT void JNICALL Java_stereolab_CUDADiff_doCalculationNative
  (JNIEnv *env, jclass clazz, jobjectArray lbJ, jobjectArray rbJ, jint b, jint h, jint tauMax,
	jboolean useS, jboolean useF, jint s, jobjectArray profileJ, jobjectArray validJ)
{

	int zeilen = env->GetArrayLength(lbJ);

	jobject zeile1 = env->GetObjectArrayElement(lbJ, 0);
	int spalten = env->GetArrayLength((_jarray*)zeile1);

	signed char *lb; // left image
	signed char *rb; // right image
	int *profile; // profile matrix. to be filled by the algorithm
	unsigned char *valid; // valid matrix. to be filled by the algorithm

	lb = (signed char *)calloc(zeilen*spalten,sizeof(char));
	rb = (signed char *)calloc(zeilen*spalten,sizeof(char));
	profile = (int *)calloc(zeilen*spalten,sizeof(int));
	valid = (unsigned char *)calloc(zeilen*spalten,sizeof(unsigned char));

	printf("Copying image from Java to C\n");
	copyImageFromJNI(env, lb, lbJ, zeilen, spalten);
	copyImageFromJNI(env, rb, rbJ, zeilen, spalten);

	printf("Starting CUDA with spalten=%i, zeilen=%i, tauMax=%i, b=%i, h=%i, useS=%i, useF=%i, s=%i\n",
			spalten, zeilen, tauMax, b,h,useS,useF,s);
	doCUDACalc(lb, rb, zeilen, spalten,tauMax, profile,
		valid, b, h, useS, useF, s);

/* Calc on Host CPU: 
	for (int i=0; i<spalten; i++) for (int j=0; j<zeilen; j++)
        {
	  int xu = -b / 2;
	  int xo = (b % 2 == 1) ? b / 2 : b / 2 - 1;
	  int yu = -h / 2;
	  int yo = (h % 2 == 1) ? h / 2 : h / 2 - 1;

	  if ((i + xu - tauMax < 0) | (i + xo + tauMax >= spalten) | (j + yu < 0) | (j + yo >= zeilen))
          {

            *(profile+j*spalten+i)=0;
            *(valid+j*spalten+i)=0;
          }
	  else
          {
            int optIndex = 0;
            int optWert = 20000000;
            int val = 0;
            for (int t = -tauMax; t <= tauMax; t++)
            {
                int sum = 0;
                for (int k = xu; k <= xo; k++) {
                        for (int l = yu; l <= yo; l++) {
                                int left =  *(lb+((j+l)*spalten)+(i+k))   & 0xFF;
                                int right = *(rb+((j+l)*spalten)+(i+k+t)) & 0xFF;
                                sum += abs(left - right);
                        }
                }
                //printf("%i", sum);
                if (sum < optWert) {
                        optWert = sum;
                        optIndex = t;
                        val = 1;
                } else if (sum == optWert) {
                        val = 0;
                }
            }
            *(profile+j*spalten+i)=optIndex;
            *(valid+j*spalten+i)=val;
          }
        }
*/
	printf("Copying results back to Java\n");
	copyBooleanMatrixToJNI(env, validJ, valid, zeilen, spalten);
	copyIntMatrixToJNI(env, profileJ, profile, zeilen, spalten);

	free(lb);
	free(rb);
	free(profile);
	free(valid);
}

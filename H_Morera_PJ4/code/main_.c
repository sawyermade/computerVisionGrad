/**************************
* Calibration	TSAI3D		  *
*													*
* Marcelo Gattass					*
* Manuel E. L. Fernandez	*
* Jul06,2006							*
**************************/

#include <stdio.h>
//#include <curses.h>
#include <math.h>
#include <string.h>
 
#include "tsai3D.h"

#define PRINT 0
#define MAX_POINTS 512


int loadModel(double* modelPoints, double* imagePoints, char* str)
{
   int n=0;
   FILE* fpi = fopen(str,"rt");
   
   if (fpi==NULL) { printf("Arq error\n"); return 1; }

   for (n=0; !feof(fpi); n++ ) {
      fscanf(fpi,"%lf %lf %lf %lf %lf",
				&modelPoints[3*n],&modelPoints[3*n +1],&modelPoints[3*n +2],
				&imagePoints[2*n],&imagePoints[2*n +1]);
   }

   fclose(fpi); 
	 n--;
   return n;
}

int main(int argc, char** argv)
{
	char *str = argv[1];
	char str2[] = "../images/";
	strcat(str2, str);


   double modelPoints[3*MAX_POINTS];
	double imagePoints[2*MAX_POINTS];
	 
   double A[3*3], K[3*4],distortion;
   int n;
   
	 struct tsai_camera_parameters CameraParameters;
	 struct tsai_calibration_constants CalibrationResults;

	 n = loadModel(modelPoints, imagePoints, str2);
	 /******************************** INIT CAMERA PARAMETERS ****************************************
	 - Parameters (Camera Model  ,Ncx,Nfx,dx,dy, Cx,Cy, sx, struct tsai_camera_parameters *cp)
	 *************************************************************************************************/

	 //can pass NEW_CAMERA and the parameters to the function

	 tsai3D_define_camera_parameters(IPHONE_7,0,0,0,0,0,0,0,&CameraParameters);
	 printf("\n****************** GET TSAI 3D CALIBRATION *****************\n");
	 if(tsai3D_calibration(n, modelPoints, imagePoints,A,K,&distortion,&CameraParameters,&CalibrationResults)){
	 	
			printf("\n\nCamera parameters for file %s:\n",str);
			printf("\nNcx :	%.3lf \t	 Nfx: %.3lf	\n",CameraParameters.Ncx,CameraParameters.Nfx);
			printf("\nCell size (dx,dy):	(%.7lf,%7lf) \n",CameraParameters.dx,CameraParameters.dx);
			printf("\nImage center (Cx,Cy): (%.3lf, %.3lf) \n",CameraParameters.Cx / 2,CameraParameters.Cy / 2);      
			printf("\nsx : %.3lf  \n\n",CameraParameters.sx);      

			printf("\nCalibration results:\n");
			printf("\nIntrinsic parameters:\n");
			printf("%.3lf %.3lf	 %.3lf \n",A[0],A[1],A[2]);
			printf("%.3lf %.3lf	 %.3lf \n",A[3],A[4],A[5]);
			printf("%.3lf %.3lf	 %.3lf \n\n",A[6],A[7],A[8]);      

			printf("\nExtrinsic parameters:\n");
			printf("%.3lf  %.3lf	%.3lf  %.3lf \n",K[0],K[1],K[2],K[3]);
			printf("%.3lf  %.3lf	%.3lf  %.3lf \n",K[4],K[5],K[6],K[7]);
			printf("%.3lf  %.3lf	%.3lf  %.3lf \n\n",K[8],K[9],K[10],K[11]);      

			printf("\nDistortion parameters: \t %.3lf ",distortion);
	 }
	 else{
		 printf("\nERROR IN CALIBRATION PROCESS");
	 };

	 printf("\n\n");
	 //getch();
   return 0;
}






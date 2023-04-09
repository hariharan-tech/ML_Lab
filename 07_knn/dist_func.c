// #include <stdio.h>
#include <math.h>

struct Point2D{
  float x,y;
};

/* 
    2-D function to calculate Euclidean distance
*/
extern float euclidean2d(struct Point2D a,struct Point2D b){
  return sqrtf(powf((a.x-b.x),2) + powf((a.y-b.y),2));
}

/* 
    2-D function to calculate Manhattan distance
*/
extern float manhattan2d(struct Point2D a,struct Point2D b){
  return fabs(a.x-b.x)+fabs(a.y-b.y);
}

/* 
    2-D function to calculate Minkowski distance
    As of now this is not used
*/
extern float minkowski2d(struct Point2D a,struct Point2D b,float powval){
  return powf(powf((a.x-b.x),powval) + powf((a.y-b.y),powval),(float)(1/powval));
}

/* 
    2-D function to calculate Supremum distance
*/
extern float supremum2d(struct Point2D a,struct Point2D b){
    if(fabs(a.x-b.x)>fabs(a.y-b.y)) return fabs(a.x-b.x);
    return fabs(a.y-b.y);
}

/* 
    2-D function to calculate Cosine Similarity distance
*/
extern float cossim2d(struct Point2D a,struct Point2D b){
    float den = (sqrtf(powf(a.x,2)+powf(b.x,2))*sqrtf(powf(a.y,2)+powf(b.y,2)));
    if (den == 0) return -2; // Cos cant be lesser than -1, so giving -2 in case the denominator is 0
    return (a.x*b.x + a.y*b.y)/den;
}

// Driver Programs
// int main(){
//   printf("%f",euclidean(3,4,0,0));
//   return 0;
// }
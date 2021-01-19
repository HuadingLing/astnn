#include<stdio.h>
#include "knn.h"

int main(){
    std::cout << "Hello World!";
    return 0;
}

int A = 1+2;
    
int mainA()
{
    int n,i,shuzu[111],count1=0,count3=0,count2=0,count4=0,count5=0,count6=0;
    scanf("%d",&n);
    while(n>=100){
                  n=n-100;
                  count1++;
                  }
                      while(n>=50){
                  n=n-50;
                  count2++;
                  }
                      while(n>=20){
                  n=n-20;
                  count3++;
                  }
                      while(n>=10){
                  n=n-10;
                  count4++;
                  }    while(n>=5){
                  n=n-5;
                  count5++;
                  }
                      while(n>=1){
                  n=n-1;
                  count6++;
                  }
               printf("%d\n%d\n%d\n%d\n%d\n%d",count1,count2,count3,count4,count5,count6);
               return 0;
               }

float B = 2/3;

int mainB()
{
	int num,j,i,an[6]={100,50,20,10,5,1};
	cin>>num;
	cout<<num/an[0]<<endl;
	for(i=1;i<6;i++)
	{
	num=num-num/an[i-1]*an[i-1];
	cout<<num/an[i]<<endl;
	}
	return 0;
}
double C = 4*5;
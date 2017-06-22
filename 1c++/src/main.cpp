void detectFace();
void detectLandmarks();
int detectAUsOld();
void recognizeFaces();

int main(int argc, char **argv) 
{
	detectFace();
	detectLandmarks();
	detectAUsOld();
	recognizeFaces();
	
	return 0;
}
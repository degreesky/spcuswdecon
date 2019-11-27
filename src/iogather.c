/*
 * iogather.c
 *
 *  Created on: Jul 31, 2019
 *      Author: mathieu
 */
#include "../headers/iogather.h"


segy *get_newgather(cwp_String *key,cwp_String *type,Value *n_val,
			int *nt,int *ntr, float *dt,int *first)
/***********************************************************************
get_newgather - get a gather from stdin upon the key gather selected.
 *           Mathieu MILENKOVIC implemented the update version so the
 *           gather can have a different number of traces independently.
 * update July 2016
************************************************************************
Input:
key		header key of ensemble
type		header value type
value		value of ensemble key word
nt		number of time samples returned
ntr		number of traces in ensemble returned
dt		time sampling interval returned
first		flag for the first emsemble=0. =1 for other sucessive gather(s)
************************************************************************
Notes:
The input seismic dataset must be sorted into ensembles defined by key
************************************************************************
Author: Mathieu MILENKOVIC - use of original verision from Potash Corporation Sascatchewan, Balasz Nemeth
given to CWP in 2008.
 *
***********************************************************************/
{
        segy tr;
	static int nsegy;
        static segy *rec=NULL;
	FILE *tracefp=NULL;
	FILE *headerfp=NULL;
        static FILE *ntracefp;
        static FILE *nheaderfp;
	int indx=0;
	static Value val;
	int CheckWriteHdr, CheckWriteTrc;
    int ntrr = 0;
    int eog = 0, eof = 0;
        //int s = 0;
	*type = hdtype(*key);
	indx = getindex(*key);
        *ntr = 0;

        tracefp = tmpfile();
	headerfp = tmpfile();

        /*eog =
         example of a gather with 6 trace per gather
        ntr++ | 1 2 3 4 5 6  1 2 3 4 5 6 |
         val  | 1 1 1 1 1 1  2 2 2 2 2 2 |
         nval | 1 1 1 1 1 2  2 2 2 2 2 0 |
        eog   | 0 0 0 0 0 1  0 0 0 0 0 1 |
        eof   | 0 0 0 0 0 0  0 0 0 0 0 1 |
         s    | 0 0 0 0 0 1  0 0 0 0 0 1 |  = eog

          example of 1 trace per gather
        ntr++ | 1 2 |
        eog   | 1 1 |
        eof   | 0 1 |
         s    | 1 1 | = eog
        */


        if(*first==0) {
            nsegy = gettr(&tr);
            *nt = tr.ns;
            *dt = (float) tr.dt/1000000.0;
            *first=1;
            ntracefp=NULL; /* first different trace pointer */
            nheaderfp=NULL;
            ntracefp = tmpfile();
            nheaderfp = tmpfile();
            ntrr = 0;

            if (nsegy==0){
                eog = 1;
                eof = 1;
                //s = 1;
                ntrr = 0;
                *ntr = ntrr;
                *nt = 0;
                *dt = 0;
                fclose(headerfp);
                fclose(tracefp);
                fclose(nheaderfp);
                fclose(ntracefp);
                //free(rec);
                err("can't get first trace");
                return(rec=NULL);
            }


        }
        else{
            eog = 0;
            eof = 0;
            //s = 0;
            if(nsegy == 0){
                ntrr = 0;
                *ntr = ntrr;
                *nt = 0;
                *dt = 0;
                //free(rec);
                return(rec=NULL);

            }
            fseek(nheaderfp,0,SEEK_SET);//move to beginning of file
            fseek(ntracefp,0,SEEK_SET);//move to beginning of file
            fread (&tr,HDRBYTES, 1, nheaderfp);
            fread (&tr.data,sizeof(float), *nt, ntracefp);
            *nt = tr.ns;
            *dt = (float) tr.dt/1000000.0;
            gethval(&tr, indx, n_val);
        }

        //at least 1 trace pass
	do {
                if((ntrr) == 0){
                    eog = 0;
                    eof = 0;
                    //s = 0;
                    gethval(&tr, indx, &val); //get current key value
                    ntrr++;
                    //warn("ntrr=%d eog=%d eof=%d s=%d\n",ntrr,eog,eof,s);

                }else{
                    fseek(headerfp, 0, SEEK_END); //move to end of file
                    fseek(tracefp, 0, SEEK_END);  //move to end of file
                    CheckWriteHdr = fwrite(&tr, 1,HDRBYTES, headerfp);              //write header to header file
                    CheckWriteTrc = fwrite(&tr.data, sizeof(float),*nt, tracefp);   //write samples to trace file
                    assert(CheckWriteHdr > 0);
                    assert(CheckWriteTrc > 0);

                    gethval(&tr, indx, &val);
                    //get new trace
                    nsegy = gettr(&tr);
                    fseek(nheaderfp,0,SEEK_SET);//move to beginning of file
                    fseek(ntracefp,0,SEEK_SET);//move to beginning of file
                    fwrite(&tr, 1,HDRBYTES, nheaderfp);
                    fwrite(&tr.data, sizeof(float),*nt, ntracefp);
                    gethval(&tr, indx, n_val);
                    //warn("val=%d nval=%d\n",val,*n_val);
                    if (nsegy ){
                        eog = 0;
                        eof = 0;
                        //s = 0;
                        if (valcmp(*type,val,*n_val)){
                            eog = 1;
                            eof = 0;
                            //s = 1;
                            fseek(nheaderfp,0,SEEK_SET);//move to beginning of file
                            fseek(ntracefp,0,SEEK_SET);//move to beginning of file
                            CheckWriteHdr = fwrite(&tr, 1,HDRBYTES, nheaderfp);              //write header to header file
                            CheckWriteTrc = fwrite(&tr.data, sizeof(float),*nt, ntracefp);   //write samples to trace file
                            break;

                        }
                    }
                    else if(!nsegy){
                        eog = 1;
                        eof = 1;
                        //s = 1;
                        break;
                    }
                    ntrr++;

                }
                //warn("ntrr=%d eog=%d eof=%d s=%d\n",ntrr,eog,eof,s);

        }while( !eog );
        /* allocate memory for the record */
        if(ntrr !=0){
            rec = (segy*)malloc(ntrr*sizeof(segy));
            register int ix;
            fseek(headerfp,0,SEEK_SET);//move to beginning of file
            fseek(tracefp,0,SEEK_SET);//move to beginning of file

            for (ix=0; ix<ntrr; ix++){
                fread(&rec[ix],HDRBYTES, 1, headerfp);
                fseek(headerfp,0,SEEK_CUR);
            }
            fclose (headerfp);
            if( eof == 1 ){fclose (nheaderfp);}

            for(ix=0; ix<ntrr; ix++){
                fread(&rec[ix].data,sizeof(float), *nt, tracefp);
                fseek(tracefp,0,SEEK_CUR);
            }
            fclose (tracefp);
            if( eof == 1 ){fclose (ntracefp);}

            *ntr = ntrr;
            *n_val=val;
        }
    return(rec);
}



void put_newgather(segy *data,int *nt, int *ntr)
/***********************************************************************
put_gather - put a gather to stdout
************************************************************************
Input:
rec		array of segy traces
nt		number of time samples per trace
ntr		number of traces in ensemble
************************************************************************
Author: Mathieu MILENKOVIC,
 * Update July 2016
***********************************************************************/
{
    segy outtr;
    register int i;
            for(i=0;i<*ntr;++i) {
            	fprintf(stderr,"i=%i\n",i);
                    memcpy(  &outtr,  &data[i], *nt*sizeof(float)+HDRBYTES);
                    puttr(&outtr);
            }
    free(data);
    data=NULL;
    return;
}

void writedata(segy* gather, int nttr, int nt, float* data, segyhdr *hdrs){
    segy outtr;
    register int intrc;
    //Write to disk or stream the gather output
	for(intrc=0;intrc<nttr;intrc++){
		//copy the original trace header
		memcpy(&outtr,  &hdrs[intrc], HDRBYTES);  // copy complete all samples within the trace
		memcpy(&outtr.data, &data[intrc*nt], sizeof(float)*nt); //Copy the samples
		//memcpy(&outtr,  gather, sizeof(segy));
		puttr(&outtr);
	}
	free(gather);
	gather=NULL;

    return;
}

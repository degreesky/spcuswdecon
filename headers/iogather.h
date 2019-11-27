/*
 * iogather.h
 *
 *  Created on: Jul 31, 2019
 *      Author: mathieu
 */

#ifndef IOGATHER_H_
#define IOGATHER_H_
/*Includes here*/
#include "su.h"
#include "segy.h"
#include "header.h"
#include "segyhdr.h"
#include <assert.h>


/*Declarations here*/
segy *get_newgather(cwp_String *key,cwp_String *type,Value *n_val, int *nt,int *ntr, float *dt, int *first);
void put_newgather(segy *data,int *nt, int *ntr);
void writedata(segy* gather, int nttr, int nt, float* data, segyhdr *hdrs);


#endif /* IOGATHER_H_ */

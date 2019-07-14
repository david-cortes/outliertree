/*************************************************************************/
/*									 */
/*  Copyright 2010 Rulequest Research Pty Ltd.				 */
/*									 */
/*  This file is part of GritBot GPL Edition, a single-threaded version	 */
/*  of GritBot release 2.01.						 */
/*									 */
/*  GritBot GPL Edition is free software: you can redistribute it	 */
/*  and/or modify it under the terms of the GNU General Public License	 */
/*  as published by the Free Software Foundation, either version 3 of	 */
/*  the License, or (at your option) any later version.			 */
/*									 */
/*  GritBot GPL Edition is distributed in the hope that it will be	 */
/*  useful, but WITHOUT ANY WARRANTY; without even the implied warranty	 */
/*  of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the	 */
/*  GNU General Public License for more details.			 */
/*									 */
/*  You should have received a copy of the GNU General Public License	 */
/*  (gpl.txt) along with GritBot GPL Edition.  If not, see		 */
/*									 */
/*      <http://www.gnu.org/licenses/>.					 */
/*									 */
/*************************************************************************/



/*************************************************************************/
/*									 */
/*		Global Data						 */
/*		-----------						 */
/*									 */
/*************************************************************************/

#include "defns.i"

Attribute	ClassAtt=0,	/* attribute to use as class */
		LabelAtt;	/* attribute to use as case ID */

int		MaxAtt,		/* max att number */
		MaxDiscrVal=3,	/* max discrete values for any att */
		MaxLabel=0,	/* max characters in case label */
		LineNo=0,	/* input line number */
		ErrMsgs=0,	/* errors found */
		AttExIn,	/* attribute exclusions/inclusions */
		TSBase=0;	/* base day for time stamps */

CaseNo		MaxCase=-1,	/* max data item number */
		LastDataCase;	/* max item in .data file */

Description	*Case=0,	/* data items */
		*SaveCase=0;	/* items in original order */

DiscrValue	*MaxAttVal=0;	/* number of values for each att */

char		*SpecialStatus=0;	/* special att treatment */

Definition	*AttDef=0;	/* definitions of implicit atts */

String		*AttName=0,	/* att names */
	  	**AttValName=0;	/* att value names */

int		VERBOSITY=0,	/* verbosity level (0 = none) */
		MAXCONDATTS=4,	/* max conditioning atts */
		MAXOUT=0;	/* max reported anoms */

Boolean		SIFT=true,	/* write sift file */
		LIST=false,	/* generate list of possible anoms */
		TargetSaved;	/* has current classatt been saved? */

CaseCount	CMINITEMS=0,	/* min group when testing contin */
		DMINITEMS=0,	/* min group when testing discr */
		SampleSize;	/* min sample size */

float		MINABNORM=8,	/* SDs for abnormal value */
		CF=50;		/* reporting level */

double		**Prior=0,	/* [att][discr value] */
		*LogCaseNo=0,	/* table of log2 */
		*Rand=0;	/* random numbers */

unsigned char	*Prec=0;
int		LastLevel=-1;	/* level of last condition saved */

char		Fn[512];	/* file name */

Boolean		*UseLogs=0,	/* use log transformation */
		*SomeMiss=0,	/* att has missing values */
		*SomeNA=0;	/* att has N/A values */

ContValue	*LowTail=0,	/* lowest value analysed */
		*HighTail=0;	/* highest ditto */

Clust		*Cluster=0;	/* clusters found */
int		NClust=0,
		ClustSpace=0;

EnvRec		GEnv;		/* global environment */

FILE		*Sf=0;		/* sift file */

Tree		T=0;		/* intermediate tree */

String		FileStem="undefined";


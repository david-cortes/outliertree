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
/*		External references to data in global.c			 */
/*		---------------------------------------			 */
/*									 */
/*************************************************************************/

extern	Attribute	ClassAtt,
			LabelAtt;

extern	int		MaxAtt,
			MaxDiscrVal,
			MaxLabel,
			LineNo,
			ErrMsgs,
			AttExIn,
			TSBase;

extern	CaseNo		MaxCase,
			LastDataCase;

extern	Description	*Case,
			*SaveCase;

extern	DiscrValue	*MaxAttVal;

extern	char		*SpecialStatus;

extern	Definition	*AttDef;

extern	String		*AttName,
		  	**AttValName;

extern	int		VERBOSITY,
			MAXCONDATTS,
			MAXOUT;

extern	Boolean		SIFT,
			LIST,
			TargetSaved;

extern	CaseCount	CMINITEMS,
			DMINITEMS,
			SampleSize;

extern	float		MINABNORM,
			CF;

extern double		**Prior,
			*LogCaseNo,
			*Rand;
extern	unsigned char	*Prec;
extern int		LastLevel;
	
extern	char		Fn[500];

extern	Boolean		*UseLogs,
			*SomeMiss,
			*SomeNA;

extern	ContValue	*LowTail,
			*HighTail;

extern	Clust		*Cluster;
extern	int		NClust,
			ClustSpace;

extern	EnvRec		GEnv;

extern	FILE		*Sf;

extern	Tree		T;

extern	String		FileStem;


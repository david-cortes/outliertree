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
/*  GritBot								 */
/*  -------								 */
/*								  	 */
/*  GritBot's function is to locate potential anomalies in data.	 */
/*  A potential anomaly is defined as a value of an attribute		 */
/*  that is "surprising" given the values of the other attributes	 */
/*  (all of which must be unsurprising).				 */
/*								  	 */
/*  Of course, this is difficult without having a model of how the	 */
/*  attribute values interrelate!  The fundamental idea behind		 */
/*  GritBot is that anomalies will show up as extreme deviations 	 */
/*  in populations of "similar" cases.  GritBot thus searches		 */
/*  for regions of the attribute space in which the values of one	 */
/*  attribute are relatively uniform, and checks for small numbers	 */
/*  of cases that have unusual values of this attribute but		 */
/*  unexceptional values of all other attributes.			 */
/*								  	 */
/*  Several parameters govern GritBot's behavior:			 */
/*								  	 */
/*    MAXFRAC	the approximate maximum percentage of outliers		 */
/*  		in any subset of cases (default 1%).			 */
/*  		The maximum number of permissible outliers in		 */
/*  		N cases is 2 SDs above the expected N * MAXFRAC.	 */
/*  		MAXFRAC must be small -- GritBot cannot find		 */
/*  		anomalies unless the data is fairly clean.		 */
/*								  	 */
/*  		Not user-adjustable.					 */
/*								  	 */
/*    MAXNORM	the maximum number of SDs by which a "normal"		 */
/*  		(non-anomalous) value can differ from the mean		 */
/*  		over the subset (default 2.67).				 */
/*								  	 */
/*  		Relevant only to continuous attributes.			 */
/*  		Not user-adjustable.					 */
/*								  	 */
/*    MINABNORM	the minimum number of SDs by which an anomaly		 */
/*  		should differ from the mean of the subset		 */
/*  		(default 8).  This value is adjusted to reflect		 */
/*  		the user-specified filtering level percent (CF):	 */
/*								  	 */
/*  		  CF=0	 -> MINABNORM=4					 */
/*  		  CF=50	 -> MINABNORM=8					 */
/*  		  CF=100 -> MINABNORM=20				 */
/*								  	 */
/*  		For discrete attributes, MINABNORM is used to		 */
/*  		set a maximum impurity threshold as			 */
/*								  	 */
/*  		  impurity / prior < 1 / (MINABNORM^2)			 */
/*								  	 */
/*  		Impurities greater than or equal to this value		 */
/*  		are not regarded as anomalies.				 */
/*								  	 */
/*    MAXCONDATTS  the maximum number of attributes that can be		 */
/*  		used to describe a subset of cases (default 4).		 */
/*  		Continuous attributes can be used to specify		 */
/*  		either a lower limit, an upper limit, or a range.	 */
/*								  	 */
/*  		User-adjustable.					 */
/*								  	 */
/*    CMINITEMS	the minimum number of cases in a subset containing	 */
/*		a potentially anomalous continuous value (value:	 */
/*		0.5% of cases or 35, whichever is greater).		 */
/*								  	 */
/*  		Not user-adjustable.					 */
/*								  	 */
/*    DMINITEMS	the minimum number of cases in a subset containing	 */
/*		a potentially anomalous discrete value (value:		 */
/*		currently same as CMINITEMS				 */
/*								  	 */
/*  		Not user-adjustable.					 */
/*								  	 */
/*								  	 */
/*  The GritBot procedure can be summarised as:				 */
/*								  	 */
/*    for each continuous attribute in turn:				 */
/*    {									 */
/*  	check whether a log transformation should be applied		 */
/*  	remove any possibly multinomial tails				 */
/*    }									 */
/*								  	 */
/*    for each attribute A in turn:					 */
/*    {									 */
/*  	remove cases with unknown values of A				 */
/*  	recursively partition the cases using attributes other		 */
/*  	than A, trying to produce maximally homogeneous subsets		 */
/*  	  * continuous attributes -- minimize variance of A		 */
/*  	  * discrete attributes -- minimize impurity of A		 */
/*  	test each subset for anomalies as above				 */
/*    }									 */
/*								  	 */
/*    report anomalies found						 */
/*									 */
/*************************************************************************/


#include "defns.i"
#include "extern.i"
#include <time.h>

#include <sys/time.h>
#include <sys/resource.h>

#define SetFOpt(V)	V = strtod(OptArg, &EndPtr);\
			if ( ! EndPtr || *EndPtr != '\00' ) break;\
			ArgOK = true
#define SetIOpt(V)	V = strtol(OptArg, &EndPtr, 10);\
			if ( ! EndPtr || *EndPtr != '\00' ) break;\
			ArgOK = true


int main(int Argc, char *Argv[])
/*  ----  */
{
    int			o;
    extern String	OptArg, Option;
    char		*EndPtr;
    Boolean		FirstTime=true, ArgOK;
    double		StartTime;
    FILE		*F;
    Attribute		Att;

    struct rlimit RL;

    /*  Make sure there is a largish runtime stack  */

    getrlimit(RLIMIT_STACK, &RL);

    RL.rlim_cur = Max(RL.rlim_cur, 16777216);

    if ( RL.rlim_max > 0 )	/* -1 if unlimited */
    {
	RL.rlim_cur = Min(RL.rlim_max, RL.rlim_cur);
    }

    setrlimit(RLIMIT_STACK, &RL);

    StartTime = ExecTime();
    PrintHeader("");

    /*  Process options  */

    while ( (o = ProcessOption(Argc, Argv, "f+v+l+c+n+srh")) )
    {
	if ( FirstTime )
	{
	    fprintf(Of, "\n    " T_Options ":\n");
	    FirstTime = false;
	}

	ArgOK = false;

	switch (o)
	{
	case 'f':   FileStem = OptArg;
		    fprintf(Of, F_Application, FileStem);
		    ArgOK = true;
		    break;
#ifdef VerbOpt
	case 'v':   SetIOpt(VERBOSITY);
		    fprintf(Of, "\tVerbosity level %d\n", VERBOSITY);
		    ArgOK = true;
		    break;
#endif
	case 'l':   SetFOpt(CF);
		    fprintf(Of, F_Filtering, CF);
		    Check(CF, 0, 100);
		    MINABNORM = ( CF < 50 ? 0.08 * CF + 4 : 0.24 * CF - 4 );
		    break;

	case 'c':   SetIOpt(MAXCONDATTS);
		    fprintf(Of, F_MaxConds, MAXCONDATTS);
		    Check(MAXCONDATTS, 0, 100);
		    break;

	case 'n':   SetIOpt(MAXOUT);
		    fprintf(Of, F_MaxOut, MAXOUT);
		    Check(MAXOUT, 1, 1000000);
		    break;

	case 's':   SIFT = false;
		    fprintf(Of, F_NoSift);
		    ArgOK = true;
		    break;

	case 'r':   LIST = true;
		    fprintf(Of, F_ListAnoms);
		    ArgOK = true;
		    break;
	}

	if ( ! ArgOK )
	{
	    if ( o != 'h' )
	    {
		fprintf(Of, F_UnrecogOpt, Option);
	    }
	    fprintf(Of, F_OptList);
	    Goodbye(1);
	}
    }

    /*  Get information on training data  */

    if ( ! (F = GetFile(".names", "r")) ) Error(NOFILE, "", "");
    GetNames(F);

    NotifyStage(READDATA);
    Progress(-1);

    if ( ! (F = GetFile(".data", "r")) ) Error(NOFILE, "", "");
    GetData(F, true);
    fprintf(Of, F_ReadData(MaxCase+1, MaxAtt, FileStem));

    LastDataCase = MaxCase;

    /*  If there is a .test file, include it too  */

    if ( (F = GetFile(".test", "r")) )
    {
	NotifyStage(READTEST);
	Progress(-1);

	GetData(F, false);

	fprintf(Of, F_ReadTest(MaxCase - LastDataCase, FileStem));
    }

    MemTrim();

    /*  Note any attribute exclusions/inclusions  */

    if ( AttExIn )
    {
	fprintf(Of, ( AttExIn == -1 ? F_AttNotChecked : F_AttChecked ));

	ForEach(Att, 1, MaxAtt)
	{
	    if ( ( StatBit(Att, SKIP) > 0 ) == ( AttExIn == -1 ) )
	    {
		fprintf(Of, "    %s\n", AttName[Att]);
	    }
	}
    }

    /*  Save a copy of original order  */

    SaveCase = Alloc(MaxCase+1, Description);
    memcpy(SaveCase, Case, (MaxCase+1) * sizeof(Description));

    CheckData();

    /*  Restore the original order  */

    Free(Case);
    Case = SaveCase;
    SaveCase = Nil;

    ReportOutliers();

    fprintf(Of, F_Time(ExecTime() - StartTime));

#ifdef	VerbOpt
    FreeDAC();
#endif

    return 0;
}

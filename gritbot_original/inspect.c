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
/*	Inspect: Use saved .sift file to check new cases.		 */
/*	-------------------------------------------------		 */
/*									 */
/*************************************************************************/


#include "defns.i"


/*************************************************************************/
/*									 */
/*		Global Data						 */
/*		-----------						 */
/*									 */
/*************************************************************************/

	Attribute	ClassAtt=0,	/* attribute to use as class */
			LabelAtt;	/* attribute to use as case ID */

	int		MaxAtt,		/* max att number */
			MaxDiscrVal=3,	/* max discrete values for any att */
			MaxLabel=0,	/* max characters in case label */
			LineNo=0,	/* input line number */
			ErrMsgs=0,	/* errors found */
			TSBase=0;	/* base day for time stamps */

	CaseNo		MaxCase=-1,	/* max data item number */
			LastDataCase,	/* max item in .data file */
			*GLp;		/* last case stack */

	Description	*Case=Nil,	/* data items */
			*SaveCase=Nil;	/* items in original order */

	DiscrValue	*MaxAttVal=Nil;	/* number of values for each att */

	char		*SpecialStatus=Nil;	/* special att treatment */

	Boolean		*UseLogs=Nil,	/* use log transformation */
			*SomeNA=Nil;	/* att has missing values */

	Definition	*AttDef=Nil;	/* definitions of implicit atts */

	String		*AttName=Nil,	/* att names */
		  	**AttValName=Nil;	/* att value names */

	Boolean		SIFT=true,	/* write sift file */
			LIST=false,	/* list case numbers */
			TargetSaved;	/* has current classatt been saved? */

	int		MAXOUT=0;	/* max anoms reported */

	float		MINABNORM=8;	/* SDs for abnormal value */

	unsigned char	*Prec=Nil;	/* [att] */
	int		LastLevel=0;
	
	Set		LeftSS=Nil;	/* temporary subset */

	float		*Surprise=Nil;	/* temporary DProb values */

	char		Fn[512];	/* file name */

	ContValue	*LowTail=Nil,	/* lowest value analysed */
			*HighTail=Nil;	/* highest ditto */

	Clust		*Cluster=Nil;	/* clusters found */
	int		NClust=0,
			ClustSpace=0;

	CaveatRec	*Caveat=Nil;	/* caveat records */
	int		NCaveat;	/* active caveats */

	EnvRec		GEnv;		/* sift environment */

	FILE		*Sf=0;		/* sift file */

	String		FileStem="undefined";


#define SetIOpt(V)	V = strtol(OptArg, &EndPtr, 10);\
			if ( ! EndPtr || *EndPtr != '\00' ) break;\
			ArgOK = true



/*************************************************************************/
/*									 */
/*	Main                                                             */
/*									 */
/*************************************************************************/


int main(int Argc, char *Argv[])
/*  ----  */
{
    CaseNo	i;
    FILE	*F;
    int	 	o;
    char	*EndPtr;
    Boolean	FirstTime=true, ArgOK;
    double	StartTime;
    extern String	OptArg, Option;

    SIFT = false;			/* important for CleanUp()! */

    StartTime = ExecTime();
    PrintHeader(" Inspector");

    /*  Process options  */

    while ( (o = ProcessOption(Argc, Argv, "f+n+rh")) )
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

	case 'n':   SetIOpt(MAXOUT);
		    fprintf(Of, F_MaxOut, MAXOUT);
		    Check(MAXOUT, 1, 1000000);
		    break;

	case 'r':   LIST = true;
		    fprintf(Of, F_ListAnoms);
		    ArgOK = true;
		    break;

	case '?':   printf("    **Unrecognised option %s\n", Option);
		    exit(1);
	}

	if ( ! ArgOK )
	{
	    if ( o != 'h' )
	    {
		fprintf(Of, F_UnrecogOpt, Option);
	    }
	    fprintf(Of, F_CkOptList);
	    exit(1);
	}
    }

    /*  Open the .sift file and recover attribute information  */

    CheckFile(".sift", false);
    fprintf(Of, F_ReadSift, FileStem);

    if ( ! (F = GetFile(".cases", "r")) ) Error(NOFILE, "", "");
    GetData(F, true);
    fprintf(Of, F_ReadCases(MaxCase+1, MaxAtt, FileStem));
    LastDataCase = MaxCase;

    /*  Remember original case order  */

    SaveCase = Alloc(MaxCase+1, Description);
    ForEach(i, 0, MaxCase)
    {
	SaveCase[i] = Case[i];
    }

    /*  Check the cases for anomalies using information recorded
	in the .sift file  */

    ProcessSift();

    /*  Restore original case order before reporting potential anomalies  */

    Free(Case);
    Case = SaveCase;
    SaveCase = Nil;

    ReportOutliers();

    fprintf(Of, F_Time(ExecTime() - StartTime));

    CleanupSift();

    return 0;
}



/*************************************************************************/
/*									 */
/*	Deallocate all dynamic storage					 */
/*									 */
/*************************************************************************/


void CleanupSift()
/*   -----------  */
{
    Attribute	Att;
    int		i, d;

    if ( Sf )
    {
	fclose(Sf);					Sf=0;
    }

    /*  Any stuff from ProcessSift()  */

    FreeUnlessNil(LeftSS);				LeftSS=Nil;
    FreeUnlessNil(Surprise);				Surprise=Nil;
    FreeUnlessNil(SomeNA);				SomeNA=Nil;
    FreeUnlessNil(GLp);					GLp=Nil;
    FreeUnlessNil(Prec);				Prec=Nil;
    FreeUnlessNil(UseLogs);				UseLogs=Nil;
    FreeUnlessNil(LowTail);				LowTail=Nil;
    FreeUnlessNil(HighTail);				HighTail=Nil;

    if ( Caveat )
    {
	ForEach(Att, 1, MaxAtt)
	{
	    FreeUnlessNil(Caveat[Att-1].Subset);
	}
	Free(Caveat);					Caveat=Nil;
    }

    Free(GEnv.Possible);
    Free(GEnv.Tested);
    ForEach(i, 0, GEnv.MaxLevel-1)
    {
	FreeUnlessNil(GEnv.Test[i].Left);
    }
    Free(GEnv.Test);

    if ( MaxCase >= 0 )
    {
	FreeData();					MaxCase=-1;
    }
    FreeUnlessNil(SaveCase);				SaveCase=Nil;

    FreeNames();

    if ( NClust > 0 )
    {
	ForEach(d, 0, NClust-1)
	{
	    FreeClust(Cluster[d]);
	}
    }
    NClust = ClustSpace = 0;
    FreeUnlessNil(Cluster);				Cluster=Nil;

    NotifyStage(0);
    Progress(0);
}



/*************************************************************************/
/*									 */
/*	Check entries in .sift file					 */
/*									 */
/*************************************************************************/


void ProcessSift()
/*   -----------  */
{
    int		E;
    Attribute	Att;

    /*  Find maximum discrete value  */

    ForEach(Att, 1, MaxAtt)
    {
	if ( Discrete(Att) && ! Exclude(Att) && MaxAttVal[Att] > MaxDiscrVal )
	{
	    MaxDiscrVal = MaxAttVal[Att];
	}
    }

    /*  Allocate variables used globally  */

    LeftSS = Alloc((MaxDiscrVal>>3)+1, unsigned char);
    Surprise = Alloc(MaxDiscrVal+1, float);
    SomeNA = Alloc(MaxAtt+1, Boolean);
    GLp  = Alloc(101, CaseNo);

    GEnv.Possible = Alloc(MaxDiscrVal+1, Boolean);
    GEnv.Tested = Alloc(MaxAtt+1, int);

    Caveat = Alloc(MaxAtt, CaveatRec);
    ForEach(Att, 1, MaxAtt)
    {
	SomeNA[Att] = true;
	Caveat[Att-1].Subset = Alloc((MaxDiscrVal>>3)+1, unsigned char);
    }

    /*  Process successive entries  */

    while ( fscanf(Sf, "%d", &E) == 1 )
    {
	switch ( E )
	{
	    case 0:
		    return;

	    case 1:
		    Case1();
		    break;

	    case 2:
		    Case2();
		    break;

	    case 3:
		    Case3();
		    break;

	    case 11:
		    Case11();
		    break;

	    case 12:
		    Case12();
		    break;

	    case 13:
		    Case13();
		    break;

	    case 21:
		    Case21();
		    break;

	    case 22:
		    Case22();
		    break;

	    default:
		    Error(BADSIFT, "entry", "");
	}
    }
}



/*************************************************************************/
/*									 */
/*	Functions for each value of the switch variable			 */
/*									 */
/*************************************************************************/


void Case1()
/*   -----  */
{
    CaseNo	i, Lp;

    if ( fscanf(Sf, "%d\n", &ClassAtt) != 1 )
    {
	Error(BADSIFT, "1", "");
    }


    /*  Remove cases with unknown value of new target and, if relevant,
	those with values in excluded tails  */

    Lp = MaxCase;
    for ( i = MaxCase ; i >= 0; i-- )
    {
	if ( Unknown(Case[i], ClassAtt) ||
	     ( Continuous(ClassAtt) &&
	       ( CVal(Case[i], ClassAtt) < LowTail[ClassAtt] ||
		 CVal(Case[i], ClassAtt) > HighTail[ClassAtt] ) ) )
	{
	    Swap(i, Lp);
	    Lp--;
	}
    }

    /*  Initialise level etc  */

    GEnv.Level = -1;
    GLp[0] = Lp;
}



void Case2()
/*   -----  */
{
    Attribute	Att;

    if ( fscanf(Sf, "%d\n", &Att) != 1 )
    {
	Error(BADSIFT, "2", "");
    }

    UseLogs[Att] = true;
}



void Case3()
/*   -----  */
{
    Attribute	Att;

    if ( fscanf(Sf, "%d", &Att) != 1 ||
	 fscanf(Sf, "%g %g\n", &LowTail[Att], &HighTail[Att]) != 2 )
    {
	Error(BADSIFT, "3", "");
    }
}



void Case11()
/*   ------  */
{
    Attribute	Att;
    DiscrValue	Br;

    if ( fscanf(Sf, "%d %d %d\n", &GEnv.Level, &Att, &Br) != 3 )
    {
	Error(BADSIFT, "11", "");
    }

    Filter(Att, Br, 0, Nil);
}



void Case12()
/*   ------  */
{
    Attribute	Att;
    DiscrValue	Br;
    float	Cut;

    if ( fscanf(Sf, "%d %d %d %g\n", &GEnv.Level, &Att, &Br, &Cut) != 4 )
    {
	Error(BADSIFT, "12", "");
    }

    Filter(Att, Br, Cut, Nil);
}



void Case13()
/*   ------  */
{
    Attribute	Att;
    DiscrValue	Br;
    int		Bytes, b, X;

    if ( fscanf(Sf, "%d %d %d", &GEnv.Level, &Att, &Br) != 3 )
    {
	Error(BADSIFT, "13", "");
    }

    Bytes = (MaxAttVal[Att]>>3) + 1;
    ForEach(b, 0, Bytes-1)
    {
	if ( fscanf(Sf, "%x", &X) != 1 )
	{
	    Error(BADSIFT, "13+", "");
	}

	LeftSS[b] = X;
    }

    Filter(Att, Br, 0, LeftSS);
}



void Case21()
/*   ------  */
{
    CaseCount	Cover, Anoms;
    CaseNo	i;
    DiscrValue	Expect, v;
    float	Frac;
    Clust	C=Nil;

    if ( fscanf(Sf, "%d %g %d", &Cover, &Frac, &Expect) != 3 )
    {
	Error(BADSIFT, "21", "");
    }

    Anoms = rint(Cover * (1 - Frac));

    ForEach(v, 1, MaxAttVal[ClassAtt])
    {
	Surprise[v] = 1;
    }

    while ( true )
    {
	if ( fscanf(Sf, "%d", &v) != 1 )
	{
	    Error(BADSIFT, "21+", "");
	}

	if ( !v ) break;

	if ( fscanf(Sf, "%g", &Surprise[v]) != 1 )
	{
	    Error(BADSIFT, "21++", "");
	}
    }

    ReadCaveats();

    ForEach(i, 0, GLp[GEnv.Level+1])
    {
	v = XDVal(Case[i], ClassAtt);

	if ( Surprise[v] < 1 && CheckCaveats(Case[i]) )
	{
	    if ( ! C )
	    {
		SetTestedAtts();
		C = NewClust(Expect, 0.0, 0.0, Anoms, Cover);
	    }

	    FoundPossibleAnom(i, C, Surprise[v]);
	}
    }
}



void Case22()
/*   ------  */
{
    CaseCount	Cover, Anoms;
    CaseNo	i;
    Clust	LowC=Nil, HighC=Nil;
    float	LowFrac, HighFrac, LowLim, HighLim, Mean, SD, Cv,
		Z, LowLimZ, HighLimZ;

    if ( fscanf(Sf, "%d %g %g %g %g %g %g",
		    &Cover, &Mean, &SD, &LowFrac, &LowLim, &HighFrac, &HighLim)
	 != 7 )
    {
	Error(BADSIFT, "22", "");
    }

    if ( UseLogs[ClassAtt] )
    {
	LowLimZ  = fabs(log(LowLim) - Mean) / SD;
	HighLimZ = fabs(log(HighLim) - Mean) / SD;
    }
    else
    {
	LowLimZ  = fabs(LowLim - Mean) / SD;
	HighLimZ = fabs(HighLim - Mean) / SD;
    }

    ReadCaveats();

    ForEach(i, 0, GLp[GEnv.Level+1])
    {
	Cv = CVal(Case[i], ClassAtt);
	if ( UseLogs[ClassAtt] ) Cv = ( Cv > 0 ? log(Cv) : -1E38 );

	Z = fabs(Mean - Cv) / SD;

	if ( LowFrac > 0 && Cv < Mean &&
	     Z - LowLimZ >= MINABNORM - MAXNORM )
	{
	    if ( CheckCaveats(Case[i]) )
	    {
		if ( ! LowC )
		{
		    SetTestedAtts();
		    Anoms = rint(Cover * (1 - LowFrac));
		    LowC = NewClust(Mean, SD, LowLim, Anoms, Cover);
		}

		FoundPossibleAnom(i, LowC, 1 / (Z * Z));
	    }
	}

	if ( HighFrac > 0 && Cv > Mean &&
	     Z - HighLimZ >= MINABNORM - MAXNORM )
	{
	    if ( CheckCaveats(Case[i]) )
	    {
		if ( ! HighC )
		{
		    SetTestedAtts();
		    Anoms = rint(Cover * (1 - HighFrac));
		    HighC = NewClust(Mean, SD, HighLim, Anoms, Cover);
		}

		FoundPossibleAnom(i, HighC, 1 / (Z * Z));
	    }
	}
    }
}



/*************************************************************************/
/*									 */
/*	Read any caveats on other attributes				 */
/*									 */
/*************************************************************************/


void ReadCaveats()
/*   -----------  */
{
    Attribute	Att;
    int		Bytes, Byte, b;

    NCaveat = 0;

    while ( getc(Sf) == ' ' )
    {
	if ( fscanf(Sf, "%d", &Att) != 1 )
	{
	    Error(BADSIFT, "caveat", "");
	}

	Caveat[NCaveat].Att = Att;

	if ( Continuous(Att) )
	{
	    if ( fscanf(Sf, "%g %g",
			    &Caveat[NCaveat].Low, &Caveat[NCaveat].High) != 2 )
	    {
		Error(BADSIFT, "caveat", "");
	    }
	}
	else
	{
	    Bytes = (MaxAttVal[Att]>>3) + 1;
	    ForEach(b, 0, Bytes-1)
	    {
		if ( fscanf(Sf, "%x", &Byte) != 1 )
		{
		    Error(BADSIFT, "caveat", "");
		}
		Caveat[NCaveat].Subset[b] = Byte;
	    }
	}

	NCaveat++;
    }
}



/*************************************************************************/
/*									 */
/*	Check that case satisfies all caveats				 */
/*									 */
/*************************************************************************/


Boolean CheckCaveats(Description Case)
/*      ------------  */
{
    Attribute	Att;
    int		j;
    ContValue	Cv;
    DiscrValue	Dv;

    for ( j = 0 ; j < NCaveat ; j++ )
    {
	Att = Caveat[j].Att;

	if ( Continuous(Att) )
	{
	    if ( ! Unknown(Case, Att) && ! NotApplic(Case, Att) &&
		 ( (Cv = CVal(Case, Att)) < Caveat[j].Low ||
		   Cv > Caveat[j].High ) )
	    {
		return false;
	    }
	}
	else
	{
	    Dv = XDVal(Case, Att);
	    if ( In(Dv, Caveat[j].Subset) ) return false;
	}
    }

    return true;
}



/*************************************************************************/
/*									 */
/*	Found one -- see whether more interesting than current		 */
/*									 */
/*************************************************************************/


void FoundPossibleAnom(CaseNo i, Clust C, float Xv)
/*   -----------------  */
{
    Clust	OldC;

    OldC = OutClust(Case[i]);

    if ( ! OldC ||
	 C->NCond < OldC->NCond ||
	 C->NCond == OldC->NCond && Xv < OutXVal(Case[i]) )
    {
	RecordOutlier(i, C, Xv);
    }
}



/*************************************************************************/
/*									 */
/*	Add test to the test stack and select relevant cases		 */
/*									 */
/*************************************************************************/


void Filter(Attribute Att, DiscrValue Br, ContValue Cut, Set Left)
/*   ------  */
{
    NoteTest(Att, Br, Cut, Left);

    GLp[GEnv.Level+1] = Group(Att, Br, 0, GLp[GEnv.Level], Cut, Left) - 1;
}



/*************************************************************************/
/*									 */
/*	Determine attributes used in all conditions			 */
/*									 */
/*************************************************************************/


void SetTestedAtts()
/*   -------------  */
{
    Attribute	Att;
    int		i;

    ForEach(Att, 1, MaxAtt)
    {
	GEnv.Tested[Att] = false;
    }

    ForEach(i, 0, GEnv.Level)
    {
	GEnv.Tested[GEnv.Test[i].Att] = true;
    }
}

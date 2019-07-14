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



#ifndef INSPECT
/*************************************************************************/
/*								 	 */
/*    Divide-and-Conquer generic routines				 */
/*    -----------------------------------				 */
/*								 	 */
/*************************************************************************/


#include "defns.i"
#include "extern.i"


/*************************************************************************/
/*								 	 */
/*	Allocate space for all tables				 	 */
/*								 	 */
/*************************************************************************/


void InitialiseDAC()
/*   -------------  */
{ 
    DiscrValue	v;
    Attribute	Att;
    CaseNo	i, MaxSampleSize;
    extern SortPair	*Pair;

    UseLogs  = AllocZero(MaxAtt+1, Boolean);
    SomeMiss = AllocZero(MaxAtt+1, Boolean);
    SomeNA   = AllocZero(MaxAtt+1, Boolean);
    LowTail  = AllocZero(MaxAtt+1, ContValue);
    HighTail = AllocZero(MaxAtt+1, ContValue);

    LogCaseNo = Alloc(MaxCase+2, double);
    LogCaseNo[0] = LogCaseNo[1] = 0;
    ForEach(i, 2, MaxCase+1)
    {
	LogCaseNo[i] = log((double) i) / Log2;
    }

    /*  Save random numbers for sampling  */

    MaxSampleSize = SAMPLEUNIT * Max(MaxDiscrVal, 5);
    if ( MaxSampleSize > (MaxCase+1) / 2 + 1 )
    {
	MaxSampleSize = (MaxCase+1) / 2 + 1;
    }

    Rand = Alloc(MaxSampleSize, double);
    ResetKR(1230);
    ForEach(i, 0, MaxSampleSize-1)
    {
	Rand[i] = KRandom();
    }

    /*  Compute prior probabilities for discrete values  */

    Prior = Alloc(MaxAtt+1, double *);
    ForEach(Att, 1, MaxAtt)
    {
	if ( Discrete(Att) )
	{
	    Prior[Att] = AllocZero(MaxAttVal[Att]+1, double);

	    ForEach(i, 0, MaxCase)
	    {
		Prior[Att][XDVal(Case[i], Att)]++;
	    }

	    SomeMiss[Att] = ( Prior[Att][0] > 0 );
	    SomeNA[Att]   = ( Prior[Att][1] > 0 );

	    ForEach(v, 0, MaxAttVal[Att])
	    {
		Prior[Att][v] /= (double) (MaxCase+1);
	    }
	}
    }

    /*  Determine precision for continuous attributes  */

    Prec = AllocZero(MaxAtt+1, unsigned char);
    ForEach(Att, 1, MaxAtt)
    {
	if ( ! Exclude(Att) && Continuous(Att) )
	{
	    Prec[Att] = log(FracBase(Att)) / log(10.0) + 0.5;
	}
    }

    Pair = Alloc(MaxCase+1, SortPair);

    InitialiseEnvData();
}



void FreeDAC()
/*   -------  */
{ 
    extern SortPair	*Pair;

    FreeEnvData();

    FreeUnlessNil(UseLogs);				UseLogs = Nil;
    FreeUnlessNil(SomeMiss);				SomeMiss = Nil;
    FreeUnlessNil(SomeNA);				SomeNA = Nil;
    FreeUnlessNil(LowTail);				LowTail = Nil;
    FreeUnlessNil(HighTail);				HighTail = Nil;
    FreeUnlessNil(LogCaseNo);				LogCaseNo = Nil;
    FreeUnlessNil(Rand);				Rand = Nil;
    FreeVector((void **) Prior, 1, MaxAtt);		Prior = Nil;
    FreeUnlessNil(Prec);				Prec = Nil;

    FreeUnlessNil(Pair);				Pair = Nil;
}



/*************************************************************************/
/*								 	 */
/*	Split cases Fp through Lp				 	 */
/*	CondAtts is the current number of conditioning attributes	 */
/*								 	 */
/*************************************************************************/


void Split(CaseNo Fp, CaseNo Lp, int CondAtts, Tree Parent, DiscrValue Br,
	   Tree *Result)
/*   -----  */
{ 
    CaseNo	i;
    CaseCount	Cases;
    DiscrValue	v;
    double	Val, Sum=0, SumSq=0;
    Attribute	Att, BestAtt;
    Tree	Node;


    *Result = Nil;

    /*  Recover info about tests to this point  */

    ForEach(Att, 1, MaxAtt)
    {
	GEnv.Tested[Att] = 0;
    }

    RecoverContext(Parent, Br);

    Cases = No(Fp, Lp);
    Verbosity(1,
	fprintf(Of, "\n<%d> %d cases %d-%d\n", GEnv.Level, Cases, Fp, Lp);
	ShowContext(Fp))

    GEnv.FRAC = 1;

    /*  Determine PSD and base information. This is only approximate, since
	missing values of the tested attributes are excluded.  */

    if ( Continuous(ClassAtt) )
    {
	if ( Cases < 2 * CMINITEMS )
	{
	    Progress(Cases);
	    return;
	}

	ForEach(i, Fp, Lp)
	{
	    Val    = CClass(Case[i]);
	    Sum   += Val;
	    SumSq += Val * Val;
	}
	GEnv.PSD = SDEstimate(Cases, Sum, SumSq);
    }
    else
    {
	if ( Cases < 2 * DMINITEMS )
	{
	    Progress(Cases);
	    return;
	}

	/*  Check for pure leaf  */

	FindClassFrequencies(Fp, Lp);

	ForEach(v, 1, MaxAttVal[ClassAtt])
	{
	    if ( GEnv.ClassFreq[v] == Cases )
	    {
		Verbosity(1, fprintf(Of, "\tpure subset\n"))
		Progress(Cases);
		return;
	    }
	}

	GEnv.BaseInfo =
	    TotalInfo(GEnv.ClassFreq, 1, MaxAttVal[ClassAtt]) / Cases;
    }

    *Result = Node = Leaf(Parent, Br);

    /*  Find the best attribute split, using sampling if the number
	of cases is at least the minimum multiple of the sample size.
	Start by collecting info on discrete attributes  */

    DiscreteAttInfo(Fp, Lp, CondAtts);

    if ( Cases > SAMPLEFACTOR * SampleSize )
    {
	ChooseSplitWithSampling(Fp, Lp, CondAtts);
    }
    else
    {
	ChooseSplit(Fp, Lp, CondAtts);
    }

    /*  Save any sift entry  */

    if ( SIFT && GEnv.SiftEntry && GEnv.SiftSize )
    {
	Node->SiftEntry = strdup(GEnv.SiftEntry);
	GEnv.SiftSize     = 0;
    }

    FindBestAtt(&BestAtt, &Val);

    /*  Decide whether to branch or not  */ 

    if ( BestAtt == None )
    { 
	Verbosity(1, fprintf(Of, "\tno sensible splits\n"))

	Progress(Cases);
    } 
    else
    {
	Verbosity(1,
	    fprintf(Of, "\tbest attribute %s", AttName[BestAtt]);
	    if ( Continuous(BestAtt) )
	    {
		fprintf(Of, " cut %.3f", GEnv.Bar[BestAtt]);
	    }
	    if ( ! Continuous(ClassAtt) )
	    {
		fprintf(Of, " val %.3f inf %.3f",
			    SplitVal(GEnv.Gain[BestAtt], GEnv.Info[BestAtt]),
			    GEnv.Info[BestAtt]);
	    }
	    fprintf(Of, " gain %.3f\n", GEnv.Gain[BestAtt]);)

	/*  Carry out the recursive divide-and-conquer  */

	Node->Tested = BestAtt;

	if ( Continuous(BestAtt) || Ordered(BestAtt) )
	{
	    Node->NodeType = BrThresh;
	    Node->Forks    = 3;
	    Node->Cut      = GEnv.Bar[BestAtt];
	}
	else
	if ( Continuous(ClassAtt) && MaxAttVal[BestAtt] > 3 )
	{
	    Node->NodeType = BrSubset;
	    Node->Forks    = 3;
	    Node->Left     = Alloc((MaxAttVal[BestAtt]>>3)+1, unsigned char);
	    memcpy(Node->Left, GEnv.Subset[BestAtt], (MaxAttVal[BestAtt]>>3)+1);
	}
	else
	{
	    Node->NodeType = BrDiscr;
	    Node->Forks    = MaxAttVal[BestAtt];
	}

	Node->Branch = Alloc(Node->Forks+1, Tree);

	if ( ! GEnv.Tested[BestAtt] ) CondAtts++;

	Divide(Node, Fp, Lp, CondAtts);
    }
} 



/*************************************************************************/
/*								 	 */
/*	Recover information on level and tests from tree and parent	 */
/*								 	 */
/*************************************************************************/


void RecoverContext(Tree T, DiscrValue Br)
/*   --------------  */
{
    if ( T )
    {
	RecoverContext(T->Parent, T->Br);

	NoteTest(T->Tested, Br, T->Cut, T->Left);
	GEnv.Tested[T->Tested]++;
	GEnv.Level++;
    }
    else
    {
	GEnv.Level = 0;
    }
}



/*************************************************************************/
/*								 	 */
/*	Analyse all discrete attributes in one pass			 */
/*								 	 */
/*************************************************************************/


void DiscreteAttInfo(CaseNo Fp, CaseNo Lp, int CondAtts)
/*   ---------------  */
{
    CaseNo	i;
    DiscrValue	v, c;
    Attribute	Att;
    double	Val;
    int		NDList=0, dl;

    /*  Initialise counts etc and prepare list of attributes  */

    ForEach(Att, 1, MaxAtt)
    {
	if ( ! Discrete(Att) || Exclude(Att) || Att == ClassAtt ||
	     CondAtts >= MAXCONDATTS && ! GEnv.Tested[Att] )
	{
	    continue;
	}

	GEnv.DList[NDList++] = Att;

	if ( Continuous(ClassAtt) )
	{
	    ForEach(v, 0, MaxAttVal[Att])
	    {
		GEnv.DValSum[Att][v] = GEnv.DValSumSq[Att][v] = 0;
		GEnv.DFreq[Att][v][0] = 0;	/* value frequency */
	    }
	}
	else
	{
	    ForEach(v, 0, MaxAttVal[Att])
	    {
		ForEach(c, 1, MaxAttVal[ClassAtt])
		{
		    GEnv.DFreq[Att][v][c] = 0;
		}
	    }
	}
    }

    if ( ! NDList-- ) return;

    /*  Examine cases and update all counts etc  */

    ForEach(i, Fp, Lp)
    {
	ForEach(dl, 0, NDList)
	{
	    Att = GEnv.DList[dl];

	    v = XDVal(Case[i], Att);

	    if ( Continuous(ClassAtt) )
	    {
		Val = CClass(Case[i]);

		GEnv.DFreq[Att][v][0]++;
		GEnv.DValSum[Att][v]   += Val;
		GEnv.DValSumSq[Att][v] += Val * Val;
	    }
	    else
	    {
		GEnv.DFreq[Att][v][ DClass(Case[i]) ]++;
	    }
	}
    }
}



/*************************************************************************/
/*								 	 */
/*	Choose split using a sample.  There are three phases:	 	 */
/*	- process discrete atts using all data				 */
/*	- for continuous atts, find gain etc from two samples and	 */
/*	  record the better value					 */
/*	- re-examine high-value continuous attributes using all cases	 */
/*								 	 */
/*************************************************************************/


void ChooseSplitWithSampling(CaseNo Fp, CaseNo Lp, int CondAtts)
/*   -----------------------  */
{
    double	Val, OldBestVal;
    Attribute	Att, BestAtt;

    /*  Process discrete attributes using all data  */

    ForEach(Att, 1, MaxAtt) 
    { 
	GEnv.Gain[Att] = None;

	if ( Exclude(Att) || Att == ClassAtt || Continuous(Att) ||
	     CondAtts >= MAXCONDATTS && ! GEnv.Tested[Att] )
	{
	    continue;
	}

	CheckSplit(Att, Fp, Lp);
    }

    /*  Process continuous attributes using two samples  */

    GEnv.FRAC = SampleSize / (double) No(Fp, Lp);

    SampleScan(Fp, Lp, CondAtts, false);
    SampleScan(Fp+SampleSize, Lp, CondAtts, true);

    GEnv.FRAC = 1;

    /*  Re-examine continuous attributes that are possible best splits
	(with value at least 70% of current best value)  */

    FindBestAtt(&BestAtt, &OldBestVal);

    if ( BestAtt != None )
    {
	Verbosity(2,
	    fprintf(Of, "      Revisit threshold %.3f (%s)\n",
			0.7 * OldBestVal, AttName[BestAtt]))

	ForEach(Att, 1, MaxAtt) 
	{ 
	    if ( Discrete(Att) || GEnv.Gain[Att] <= Epsilon ) continue;

	    Val = SplitVal(GEnv.Gain[Att], GEnv.Info[Att]);

	    GEnv.Gain[Att] = None;

	    if ( Val > 0.7 * OldBestVal )
	    {
		CheckSplit(Att, Fp, Lp);
	    }
	} 
    }
}



/*************************************************************************/
/*								 	 */
/*	Estimate Gain etc of continuous attributes using sample		 */
/*								 	 */
/*************************************************************************/


void SampleScan(CaseNo Fp, CaseNo Lp, int CondAtts, Boolean Second)
/*   ----------  */
{
    CaseNo	i, SLp;
    double	Val, Sum=0, SumSq=0, SaveBaseInfo, SavePSD, 
		FBar, FInfo, FGain, FVal;
    Attribute	Att;

    /*  Save base information or SD  */

    SaveBaseInfo = GEnv.BaseInfo;
    SavePSD      = GEnv.PSD;

    /*  Generate sample in Fp ... Fp+SampleSize-1  */

    Sample(Fp, Lp, SampleSize);
    SLp  = Fp + SampleSize - 1;

    /*  Determine sample PSD or base information  */

    if ( Continuous(ClassAtt) )
    {
	ForEach(i, Fp, SLp)
	{
	    Val    = CClass(Case[i]);
	    Sum   += Val;
	    SumSq += Val * Val;
	}
	GEnv.PSD = SDEstimate(SampleSize, Sum, SumSq);
    }
    else
    {
	FindClassFrequencies(Fp, SLp);
	GEnv.BaseInfo =
	    TotalInfo(GEnv.ClassFreq, 1, MaxAttVal[ClassAtt]) / SampleSize;
    }

    /*  Check attributes using sample  */

    ForEach(Att, 1, MaxAtt) 
    { 
	if ( Exclude(Att) || Att == ClassAtt || ! Continuous(Att) ||
	     CondAtts >= MAXCONDATTS && ! GEnv.Tested[Att] )
	{
	    continue;
	}

	/*  Save information from possible earlier sample  */

	FInfo = GEnv.Info[Att];
	FGain = GEnv.Gain[Att];
	FBar  = GEnv.Bar[Att];

	GEnv.Gain[Att] = None;

	CheckSplit(Att, Fp, SLp);

	/*  If this is second sample, retain information from better  */

	if ( Second )
	{
	    FVal = SplitVal(FGain, FInfo);		 /* first value */
	    Val  = SplitVal(GEnv.Gain[Att], GEnv.Info[Att]); /* second value */

	    if ( FVal > Val )
	    {
		GEnv.Gain[Att] = FGain;
		GEnv.Info[Att] = FInfo;
		GEnv.Bar[Att]  = FBar;
	    }
	}
    }

    /*  Restore base information or SD  */

    GEnv.BaseInfo = SaveBaseInfo;
    GEnv.PSD      = SavePSD;
}



/*************************************************************************/
/*								 	 */
/*	Sample N cases from Fp through Lp using tabulated random nos	 */
/*								 	 */
/*************************************************************************/


void Sample(CaseNo Fp, CaseNo Lp, CaseCount N)
/*   ------  */
{
    CaseNo	i, j, Cases;

    Cases = No(Fp, Lp);

    ForEach(i, 0, N-1)
    {
	j = Rand[i] * Cases--;
	Swap(Fp+i, Fp+j);
    }
}



/*************************************************************************/
/*								 	 */
/*	Choose a split using all cases					 */
/*								 	 */
/*************************************************************************/


void ChooseSplit(CaseNo Fp, CaseNo Lp, int CondAtts)
/*   -----------  */
{
    Attribute	Att;

    GEnv.FRAC = 1;

    ForEach(Att, 1, MaxAtt) 
    { 
	GEnv.Gain[Att] = None;

	if ( Exclude(Att) || Att == ClassAtt ||
	     CondAtts >= MAXCONDATTS && ! GEnv.Tested[Att] )
	{
	    continue;
	}

	CheckSplit(Att, Fp, Lp);
    } 
}



void FindBestAtt(Attribute *BestAtt, double *BestVal)
/*   -----------  */
{
    Attribute	Att;
    double	Val;

    *BestVal = Epsilon;
    *BestAtt = None;

    ForEach(Att, 1, MaxAtt)
    {
	Val = SplitVal(GEnv.Gain[Att], GEnv.Info[Att]);

	if ( Val > *BestVal )
	{ 
	    *BestAtt = Att; 
	    *BestVal = Val;
	} 
    }
}



/*************************************************************************/
/*								 	 */
/*	Evaluate a potential split					 */
/*								 	 */
/*************************************************************************/


void CheckSplit(Attribute Att, CaseNo Fp, CaseNo Lp)
/*   ----------  */
{
    CaseNo	Xp;

    GEnv.Tested[Att]++;

    /*  Remove missing values of Att.  Note: this makes values
	of BaseInfo and PSD approximate only  */

    Xp = ( SomeMiss[Att] ? SkipMissing(Att, Fp, Lp) : Fp );

    /*  Evaluate attribute for split -- different methods for
	continuous and discrete class attributes  */

    if ( Continuous(Att) )			/* continuous att */
    {
	if ( Continuous(ClassAtt) )
	{
	    CEvalContinAtt(Att, Xp, Lp);
	}
	else
	{
	    DEvalContinAtt(Att, Xp, Lp);
	}
    }
    else					/* discrete att */
    {
	if ( Continuous(ClassAtt) )
	{
	    if ( MaxAttVal[Att] > 3 || GEnv.Tested[Att] <= 1 )
	    {
		CEvalDiscrAtt(Att, Xp, Lp);
	    }
	}
	else
	if ( Ordered(Att) )
	{
	    DEvalOrderedAtt(Att, Xp, Lp);
	}
	else
	if ( GEnv.Tested[Att] <= 1 )
	{
	    DEvalDiscrAtt(Att, Xp, Lp);
	}
    }

    if ( GEnv.Gain[Att] > Epsilon )
    {
	/*  Find value adjusted for missing values  */

	GEnv.Gain[Att] *= No(Xp, Lp) / (double) No(Fp, Lp);
	GEnv.Info[Att]  = Max(GEnv.Info[Att], 0.5);
    }

    GEnv.Tested[Att]--;
}



/*************************************************************************/
/*								 	 */
/*	Split cases Fp to Lp on attribute Att				 */
/*								 	 */
/*************************************************************************/


void Divide(Tree Node, CaseNo Fp, CaseNo Lp, int CondAtts)
/*   ------  */
{
    CaseNo	Ep;
    DiscrValue	v;

    /*  Remove unknown attribute values  */

    Ep = ( SomeMiss[Node->Tested] ? SkipMissing(Node->Tested, Fp, Lp) : Fp );
    Progress(Ep - Fp);

    /*  Recursive divide and conquer  */

    ForEach(v, 1, Node->Forks)
    {
	Fp = Ep;
	Ep = Group(Node->Tested, v, Fp, Lp, Node->Cut, Node->Left);

	if ( Ep > Fp )
	{
	    Split(Fp, Ep-1, CondAtts, Node, v, &Node->Branch[v]);
	}
    }
}



/*************************************************************************/
/*								 	 */
/*	Group together missing values and return index of next case	 */
/*								 	 */
/*************************************************************************/


CaseNo SkipMissing(Attribute Att, CaseNo Fp, CaseNo Lp)
/*     -----------  */
{
    CaseNo	i;

    ForEach(i, Fp, Lp)
    {
	if ( Unknown(Case[i], Att) )
	{
	    Swap(Fp, i);
	    Fp++;
	}
    }

    return Fp;
}




/*************************************************************************/
/*								  	 */
/*	Check groups formed by a potential test				 */
/*								  	 */
/*************************************************************************/


void CheckPotentialClusters(Attribute Att, DiscrValue Forks,
			    CaseNo Fp, CaseNo Lp, ContValue Cut, Set S,
			    CaseNo **FT)
/*   ----------------------  */
{
    CaseNo	Ep;
    DiscrValue	v;

    ForEach(v, 1, Forks)
    {
	Ep = Group(Att, v, Fp, Lp, Cut, S);

	if ( Ep > Fp )
	{
	    NoteTest(Att, v, Cut, S);

	    if ( Continuous(ClassAtt) )
	    {
		FindContinOutliers(Fp, Ep-1, false);
	    }
	    else
	    {
		FindDiscrOutliers(Fp, Ep-1, ( FT ? FT[v] : Nil ));
	    }

	    Fp = Ep;
	}
    }
}



/*************************************************************************/
/*								  	 */
/*	Print context information for DAC				 */
/*								  	 */
/*************************************************************************/


void ShowContext(CaseNo i)
/*   -----------  */
{
    Attribute Att;
    ClustRec	CR;
    Clust	C=&CR;
    int		d;

    C->Att = ClassAtt;
    GEnv.Level--;
    SaveClustConds(C);
    GEnv.Level++;

    ForEach(d, 0, C->NCond-1)
    {
	Att = C->Cond[d].Att;

	if ( Continuous(Att) )
	{
	    PrintContinCond(Att, C->Cond[d].Low, C->Cond[d].High, i);
	}
	else
	if (  Ordered(Att) )
	{
	    PrintOrderedCond(Att, (int) C->Cond[d].Low, (int) C->Cond[d].High,
			     i);
	}
	else
	if ( Continuous(C->Att) && MaxAttVal[Att] > 3 )
	{
	    PrintSubsetCond(Att, C->Cond[d].Values, i);
	    FreeUnlessNil(C->Cond[d].Values);
	}
	else
	{
	    PrintValCond(Att, (int) C->Cond[d].Low);
	}
    }

    Free(C->Cond);
}



/*************************************************************************/
/*									 */
/*	Construct a leaf in a given node				 */
/*									 */
/*************************************************************************/


Tree Leaf(Tree Parent, DiscrValue Br)
/*   ----  */
{
    Tree	Node;

    Node = AllocZero(1, TreeRec);

    Node->NodeType	= 0; 
    Node->Parent	= Parent;
    Node->Br		= Br;

    return Node; 
}



void ReleaseTree(Tree T, int Level)
/*   -----------  */
{
    DiscrValue	v;

    if ( ! T ) return;

    if ( Level > 0 && LastLevel >= Level - 1 ) LastLevel = Level - 2;

    /*  Possible sift entry  */

    if ( T->SiftEntry )
    {
	if ( SIFT )
	{
	    RecoverContext(T->Parent, T->Br);
	    OutputConditions();
	    fprintf(Sf, "%s", T->SiftEntry);
	}
	Free(T->SiftEntry);
    }

    if ( T->NodeType )
    {
	ForEach(v, 1, T->Forks)
	{
	    ReleaseTree(T->Branch[v], Level+1);
	}

	if ( T->NodeType == BrSubset )
	{
	    FreeUnlessNil(T->Left);
	}

	Free(T->Branch);
    }

    Free(T);
}



void OutputConditions()
/*   ----------------  */
{
    Attribute	Att;
    int		i, CType, b, Bytes;
    DiscrValue	Br;

    if ( ! TargetSaved )
    {
	fprintf(Sf, "1 %d\n", ClassAtt);
	TargetSaved = true;
    }

    if ( GEnv.Level < 0 ) return;

    /*  Save all conditions since last saved  */

    ForEach(i, LastLevel+1, GEnv.Level-1)
    {
	Att = GEnv.Test[i].Att;
	Br  = GEnv.Test[i].Br;

	/*  Determine condition type  */

	CType = ( Br == 1 ? 11 :
		  Continuous(Att) || Ordered(Att) ? 12 :
		  Continuous(ClassAtt) && MaxAttVal[Att] > 3 ? 13 : 11 );

	fprintf(Sf, "%d %d %d %d", CType, i, Att, Br);

	/*  Don't need to save anything else if this branch is 1 (N/A)
	    or if test is on two-valued discrete att  */

	if ( Br != 1 )
	{
	    if ( Continuous(Att) || Ordered(Att) )
	    {
		fprintf(Sf, " %.8g", GEnv.Test[i].Cut);
	    }
	    else
	    if ( Continuous(ClassAtt) && MaxAttVal[Att] > 3 )
	    {
		/*  Print subset of values  */

		Bytes = (MaxAttVal[Att]>>3) + 1;

		ForEach(b, 0, Bytes-1)
		{
		    fprintf(Sf, " %x", GEnv.Test[i].Left[b]);
		}
	    }
	}

	fprintf(Sf, "\n");
    }

    LastLevel = GEnv.Level-1;
}



/*************************************************************************/
/*								 	 */
/*	Set up environment						 */
/*								 	 */
/*************************************************************************/


void InitialiseEnvData()
/*   -----------------  */
{
    DiscrValue	v;
    Attribute	Att;

    GEnv.ValFreq   = Alloc(MaxDiscrVal+1, CaseCount);
    GEnv.ClassFreq = Alloc(MaxDiscrVal+1, CaseCount);
    GEnv.ValSum    = Alloc(MaxDiscrVal+1, double);
    GEnv.ValSumSq  = Alloc(MaxDiscrVal+1, double);
    GEnv.Left      = Alloc(MaxDiscrVal+1, Boolean);
    GEnv.Possible  = Alloc(MaxDiscrVal+1, Boolean);
    GEnv.Tested    = AllocZero(MaxAtt+1, int);
    GEnv.Gain      = AllocZero(MaxAtt+1, double);
    GEnv.Info      = AllocZero(MaxAtt+1, double);
    GEnv.Bar       = AllocZero(MaxAtt+1, ContValue);

    GEnv.Subset    = AllocZero(MaxAtt+1, Set);
    GEnv.Subset[0] = Alloc((MaxDiscrVal>>3)+1, unsigned char); /* caveats */
    ForEach(Att, 1, MaxAtt)
    {
	if ( Discrete(Att) )
	{
	    GEnv.Subset[Att] = Alloc((MaxAttVal[Att]>>3)+1, unsigned char);
	}
    }

    /*  Freq[] is one longer than apparently necessary to allow for
	the extra slot needed by EvalOrderedAtt()  */

    GEnv.Freq = AllocZero(MaxDiscrVal+2, CaseCount *);
    ForEach(v, 0, MaxDiscrVal+1)
    {
	GEnv.Freq[v] = AllocZero(MaxDiscrVal+1, CaseCount);
    }

    GEnv.BestFreq = AllocZero(4, CaseCount *);
    ForEach(v, 0, 3)
    {
	GEnv.BestFreq[v] = AllocZero(MaxDiscrVal+1, CaseCount);
    }

    GEnv.DList     = Alloc(MaxAtt+1, Attribute);
    GEnv.DFreq     = Alloc(MaxAtt+1, CaseCount **);
    GEnv.DValSum   = Alloc(MaxAtt+1, double *);
    GEnv.DValSumSq = Alloc(MaxAtt+1, double *);
    ForEach(Att, 1, MaxAtt)
    {
	if ( Exclude(Att) || ! Discrete(Att) ) continue;

	GEnv.DFreq[Att] = Alloc(MaxAttVal[Att]+1, CaseCount *);
	ForEach(v, 0, MaxAttVal[Att])
	{
	    GEnv.DFreq[Att][v] = Alloc(MaxDiscrVal+1, CaseCount);
	}
	GEnv.DValSum[Att]   = Alloc(MaxDiscrVal+1, double);
	GEnv.DValSumSq[Att] = Alloc(MaxDiscrVal+1, double);
    }
}



/*************************************************************************/
/*								 	 */
/*	Clean up environment						 */
/*								 	 */
/*************************************************************************/


void FreeEnvData()
/*   ------------  */
{ 
    Attribute	Att;
    int		i;

    if ( ! GEnv.ValFreq ) return;

    FreeUnlessNil(GEnv.ValFreq);
    FreeUnlessNil(GEnv.ClassFreq);
    FreeUnlessNil(GEnv.ValSum);
    FreeUnlessNil(GEnv.ValSumSq);
    FreeUnlessNil(GEnv.Left);
    FreeUnlessNil(GEnv.Possible);
    FreeUnlessNil(GEnv.Tested);
    FreeUnlessNil(GEnv.Gain);
    FreeUnlessNil(GEnv.Info);
    FreeUnlessNil(GEnv.Bar);

    if ( GEnv.Test )
    {
	ForEach(i, 0, GEnv.MaxLevel-1)
	{
	    FreeUnlessNil(GEnv.Test[i].Left);
	}

	Free(GEnv.Test);
    }

    FreeUnlessNil(GEnv.Subset[0]);
    ForEach(Att, 1, MaxAtt)
    {
	if ( Discrete(Att) )
	{
	    FreeUnlessNil(GEnv.Subset[Att]);
	}
    }
    FreeUnlessNil(GEnv.Subset);

    FreeVector((void **) GEnv.Freq, 0, MaxDiscrVal+1);
    FreeVector((void **) GEnv.BestFreq, 0, 3);

    ForEach(Att, 1, MaxAtt)
    {
	if ( ! GEnv.DFreq[Att] ) continue;

	FreeVector((void **)GEnv.DFreq[Att], 0, MaxAttVal[Att]);
	Free(GEnv.DValSum[Att]);
	Free(GEnv.DValSumSq[Att]);
    }
    Free(GEnv.DFreq);
    Free(GEnv.DValSum);
    Free(GEnv.DValSumSq);
    Free(GEnv.DList);

    FreeUnlessNil(GEnv.SiftEntry);
}
#endif



/*************************************************************************/
/*								 	 */
/*	Test[] contains a stack of current tests.  Add a new test	 */
/*	for the current level						 */
/*								 	 */
/*************************************************************************/


void NoteTest(Attribute Att, DiscrValue Br, ContValue Cut, Set Left)
/*   --------  */
{
    int i;

    /*  Check space for tests  */

    if ( GEnv.Level >= GEnv.MaxLevel )
    {
	if ( ! GEnv.MaxLevel )
	{
	    GEnv.Test = Alloc(100, TestRec);
	}
	else
	{
	    Realloc(GEnv.Test, GEnv.MaxLevel+100, TestRec);
	}

	ForEach(i, 0, 99)
	{
	    GEnv.Test[GEnv.MaxLevel+i].Left =
		Alloc((MaxDiscrVal>>3)+1, unsigned char);
	}

	GEnv.MaxLevel += 100;
    }

    GEnv.Test[GEnv.Level].Att = Att;
    GEnv.Test[GEnv.Level].Br  = Br;
    GEnv.Test[GEnv.Level].Cut = Cut;
    if ( Left )
    {
	memcpy(GEnv.Test[GEnv.Level].Left, Left, (MaxAttVal[Att]>>3)+1);
    }
}



/*************************************************************************/
/*								 	 */
/*	Group together the cases corresponding to branch V of a test 	 */
/*	and return the index of the case following the last	 	 */
/*								 	 */
/*************************************************************************/


CaseNo Group(Attribute Att, DiscrValue V, CaseNo Fp, CaseNo Lp,
	     ContValue Cut, Set Left)
/*     -----  */
{
    CaseNo	i;

    /*  Group cases on the value of attribute Att, perhaps depending
	on the type of split  */

    if ( V == 1 )
    {
	/*  Group all non-applicable values.  Don't even try if
	    this attribute doesn't have N/A values  */

	if ( SomeNA[Att] )
	{
	    ForEach(i, Fp, Lp)
	    {
		if ( NotApplic(Case[i], Att) )
		{
		    Swap(Fp, i);
		    Fp++;
		}
	    }
	}
    }
    else
    if ( Continuous(Att) )
    {
	ForEach(i, Fp, Lp)
	{
	    if ( ! Unknown(Case[i], Att) &&
		 ! NotApplic(Case[i], Att) &&
		 (CVal(Case[i], Att) <= Cut) == (V == 2) )
	    {
		Swap(Fp, i);
		Fp++;
	    }
	}
    }
    else
    if ( Ordered(Att) && Att != ClassAtt )
    {
	ForEach(i, Fp, Lp)
	{
	    if ( ! Unknown(Case[i], Att) &&
		 ! NotApplic(Case[i], Att) &&
		 (XDVal(Case[i], Att) <= Cut + 0.1) == (V == 2) )
	    {
		Swap(Fp, i);
		Fp++;
	    }
	}
    }
    else
    if ( Continuous(ClassAtt) && MaxAttVal[Att] > 3 )
    {
	ForEach(i, Fp, Lp)
	{
	    if ( ! Unknown(Case[i], Att) &&
		 ! NotApplic(Case[i], Att) &&
		 (In(XDVal(Case[i], Att), Left) != 0) == (V == 2) )
	    {
		Swap(Fp, i);
		Fp++;
	    }
	}
    }
    else
    {
	ForEach(i, Fp, Lp)
	{
	    if ( XDVal(Case[i], Att) == V )
	    {
		Swap(Fp, i);
		Fp++;
	    }
	}
    }

    return Fp;
}

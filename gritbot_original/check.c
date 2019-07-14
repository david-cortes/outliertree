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
/*	Principal routines to check data				 */
/*	--------------------------------				 */
/*									 */
/*************************************************************************/


#include "defns.i"
#include "extern.i"

Boolean	CheckMsg;		/* message printed for current attribute */
CaseNo	LowFp, LowLp, HighFp;	/* pointers to omitted low/high tails */



/*************************************************************************/
/*									 */
/*	Check each attribute in turn					 */
/*									 */
/*************************************************************************/


void CheckData()
/*   ---------  */
{
    CaseNo	i, Fp, Lp;

    NotifyStage(PRELIM);

    /*  Set up tables, determine discrete value priors, etc.  */

    InitialiseDAC();

    /*  Initialise random numbers for sampling  */

    ResetKR(0);

    DMINITEMS = CMINITEMS = Max(35, 0.005 * (MaxCase+1));

    LowFp  = 0;
    LowLp  = -1;
    HighFp = MaxCase+1;

    if ( SIFT )
    {
	CheckFile(".sift", true);
    }

    /*  Set ClassAtt to each attribute in turn.
	Check distribution type, remove tails, and perform global check  */

    ForEach(ClassAtt, 1, MaxAtt)
    {
	if ( Exclude(ClassAtt) )
	{
	    continue;
	}

	Verbosity(1, fprintf(Of, "\n==========\n%s\n", AttName[ClassAtt]))

	Progress(-ClassAtt);
	CheckMsg = false;

	/*  Delete missing values and set SomeMiss[].
	    SomeNA[] is set in CheckContin (for continuous atts)
	    and in InitialiseDAC (for discrete atts)  */

	Fp = SkipMissing(ClassAtt, 0, MaxCase);
	if ( (SomeMiss[ClassAtt] = ( Fp > 0 )) && ! Skip(ClassAtt) )
	{
	    fprintf(Of, F_WhileCheck, AttName[ClassAtt]);
	    fprintf(Of, F_ExcludeMissing(Fp));
	    CheckMsg = true;
	}

	if ( Fp < MaxCase )
	{
	    TargetSaved = false;
	    GEnv.Level  = -1;

	    if ( Continuous(ClassAtt) )
	    {
		CheckContin(Fp);
	    }
	    else
	    if ( SIFT )
	    {
		/*  Check for possible entries for sift file  */

		ForEach(i, Fp, MaxCase)
		{
		    Case[i][0] = Case[i][ClassAtt];
		}

		FindDiscrOutliers(Fp, MaxCase, Nil);
	    }

	    /*  Dump any sift entries  */

	    if ( SIFT && GEnv.SiftSize )
	    {
		fprintf(Sf, "1 %d\n%s", ClassAtt, GEnv.SiftEntry);
		GEnv.SiftSize = 0;
	    }
	}
    }


    /*  Search for subsets and test  */

    NotifyStage(CHECKING);

    ForEach(ClassAtt, 1, MaxAtt)
    {
	if ( Skip(ClassAtt) ) continue;

	Verbosity(1, fprintf(Of, "\n==========\n%s\n", AttName[ClassAtt]))

	Progress(-ClassAtt);

	/*  Restore original order  */

	memcpy(Case, SaveCase, (MaxCase+1) * sizeof(Description));

	/*  Remove missing values and tails of continuous attributes  */

	Fp = ( SomeMiss[ClassAtt] ? SkipMissing(ClassAtt, 0, MaxCase) : 0 );
	Lp = MaxCase;

	if ( Continuous(ClassAtt) )
	{
	    /*  Remove N/A values  */

	    if ( SomeNA[ClassAtt] ) Fp = Group(ClassAtt, 1, Fp, Lp, 0, Nil);

	    /*  Put low tails low and high tails high  */

	    LowFp = Fp;

	    ForEach(i, Fp, Lp)
	    {
		if ( CVal(Case[i], ClassAtt) < LowTail[ClassAtt] )
		{
		    Swap(i, Fp);
		    Fp++;
		}
		else
		if ( CVal(Case[i], ClassAtt) > HighTail[ClassAtt] )
		{
		    Swap(i, Lp);
		    Lp--;
		    i--;
		}
	    }

	    LowLp  = Fp-1;
	    HighFp = Lp+1;
	}

	if ( Fp > 0 ) Progress(Fp);

	/*  Copy class values  */

	if ( Continuous(ClassAtt) && UseLogs[ClassAtt] )
	{
	    ForEach(i, Fp, Lp)
	    {
		CClass(Case[i]) = log(CVal(Case[i], ClassAtt));
	    }
	}
	else
	{
	    ForEach(i, Fp, Lp)
	    {
		Case[i][0] = Case[i][ClassAtt];
	    }
	}

	SampleSize = SAMPLEUNIT *
		     ( Continuous(ClassAtt) ? 5 :
		       SomeNA[ClassAtt] ?  MaxAttVal[ClassAtt]  :
					   MaxAttVal[ClassAtt] - 1 );
	Split(Fp, Lp, 0, Nil, 0, &T);

	TargetSaved = false;
	LastLevel   = -1;

	ReleaseTree(T, 0);
	T = Nil;
    }

    if ( SIFT )
    {
	fprintf(Sf, "0\n");
	fclose(Sf);
	Sf = 0;
    }
}



/*************************************************************************/
/*									 */
/*	Check a continuous attribute					 */
/*	  -  decide whether to apply the log transformation		 */
/*	  -  exclude possibly multimodal tails				 */
/*	  -  check for global outliers					 */
/*									 */
/*************************************************************************/


void CheckContin(CaseNo Fp)
/*   -----------  */
{
    CaseNo	i, Mid, Quart, Tail, Tp, Lp;
    CaseCount	Cases, Middle;
    double	R1, R2, Mean, SD, Sum=0, SumSq=0, Cv;
    char	CVS[20];
    Boolean	LowT=false, HighT=false;

    /*  First discard any non-applicable values  */

    Tp = Fp;
    ForEach(i, Fp, MaxCase)
    {
	if ( NotApplic(Case[i], ClassAtt) )
	{
	    Swap(Fp, i);
	    Fp++;
	}
    }

    /*  Remember whether there were any  */

    if ( (SomeNA[ClassAtt] = ( Fp > Tp )) && ! Skip(ClassAtt) )
    {
	if ( ! CheckMsg )
	{
	    fprintf(Of, F_WhileCheck, AttName[ClassAtt]);
	    CheckMsg = true;
	}
	fprintf(Of, F_ExcludeNA(Fp - Tp));
    }

    if ( Fp > MaxCase - CMINITEMS ) return;

    Quicksort(Fp, MaxCase, ClassAtt);

    Mid = (MaxCase + Fp) / 2;

    /*  Check for asymmetry and potential log distribution  */

    Quart = No(Fp, MaxCase) / 4;

    if ( CVal(Case[Fp], ClassAtt) > Epsilon &&
	 CVal(Case[MaxCase-Quart], ClassAtt) > CVal(Case[Mid], ClassAtt) )
    {
	/*  R1 is (log(q2)-log(q1)) / (log(q3)-log(q2))
	    R2 is (q2-q1) / (q3-q2)
	    Choose the log distribution if R2 < 1 and R1 is closer
	    to 1 than R2  */

	R1 = ( log(CVal(Case[Mid], ClassAtt)) -
	       log(CVal(Case[Fp+Quart], ClassAtt)) ) /
	     ( log(CVal(Case[MaxCase-Quart], ClassAtt)) -
	       log(CVal(Case[Mid], ClassAtt)) );
	R2 = (CVal(Case[Mid], ClassAtt) - CVal(Case[Fp+Quart], ClassAtt)) /
	     (CVal(Case[MaxCase-Quart], ClassAtt) - CVal(Case[Mid], ClassAtt));

	UseLogs[ClassAtt] = R2 < 1 && fabs(R1-1) < fabs(R2-1);
	if ( UseLogs[ClassAtt] )
	{
	    Verbosity(1, fprintf(Of, "    Using log distribution\n"))

	    if ( SIFT )
	    {
		fprintf(Sf, "2 %d\n", ClassAtt);
	    }
	}
    }
    else
    {
	UseLogs[ClassAtt] = false;
    }

    /*  That's all that needs to be done for non-included attributes  */

    if ( Skip(ClassAtt) ) return;

    /*  Load the appropriate values into the class  */

    if ( UseLogs[ClassAtt] )
    {
	ForEach(i, Fp, MaxCase)
	{
	    CClass(Case[i]) = log(CVal(Case[i], ClassAtt));
	}
    }
    else
    {
	ForEach(i, Fp, MaxCase)
	{
	    CClass(Case[i]) = CVal(Case[i], ClassAtt);
	}
    }

    /*  Check for multimodal tails and exclude  */

    Lp    = MaxCase;
    Cases = No(Fp, Lp);
    Tail  = MaxAnoms(Cases);

    /*  Estimate SD from the central half of the data and adjust; if
	this is impossible (too many repeated values), mark the
	attribute as skipped  */

    if ( CClass(Case[Fp + Quart + Tail]) < CClass(Case[Lp - Quart - Tail]) )
    {
	ForEach(i, Fp + Quart, Lp - Quart)
	{
	    Sum += (Cv = CClass(Case[i]));
	    SumSq += Cv * Cv;
	}
	Mean = Sum / (Middle = No(Fp, Lp) - 2 * Quart);
	SD = 2.5 * SDEstimate(Middle, Sum, SumSq);
    }
    else
    {
	/*  This is not really a continuous distribution -- at least
	    half of the cases have identical values.  Skip it  */

	if ( ! CheckMsg )
	{
	    fprintf(Of, F_WhileCheck, AttName[ClassAtt]);
	    CheckMsg = true;
	}

	fprintf(Of, F_TooManyIdentical);
	SpecialStatus[ClassAtt] |= SKIP;
	return;
    }

    /*  Look for multimodal low tail  */

    for ( Tp = Fp ; Tp < Mid && ZScore(Tp) >= MAXTAIL ; Tp++ )
	;

    if ( Tp - Fp > Tail )
    {
	if ( ! CheckMsg )
	{
	    fprintf(Of, F_WhileCheck, AttName[ClassAtt]);
	    CheckMsg = true;
	}

	CValToStr(CVal(Case[Tp], ClassAtt), ClassAtt, CVS);
	fprintf(Of, F_LowTail(Tp - Fp, CVS));
	Fp = Tp;
	LowT = true;
    }

    /*  Ditto multimodal high tail  */

    for ( Tp = Lp ; Tp > Mid && ZScore(Tp) >= MAXTAIL ; Tp-- )
	;

    if ( Lp - Tp > Tail )
    {
	if ( ! CheckMsg )
	{
	    fprintf(Of, F_WhileCheck, AttName[ClassAtt]);
	    CheckMsg = true;
	}

	CValToStr(CVal(Case[Tp], ClassAtt), ClassAtt, CVS);
	fprintf(Of, F_HighTail(Lp - Tp, CVS));
	Lp = Tp;
	HighT = true;
    }

    /*  Record tail information  */

    LowTail[ClassAtt]  = CVal(Case[Fp], ClassAtt);
    HighTail[ClassAtt] = CVal(Case[Lp], ClassAtt);

    if ( SIFT && ( LowT || HighT ) )
    {
	fprintf(Sf, "3 %d %.8g %.8g\n", ClassAtt,
		    ( LowT ? CVal(Case[Fp], ClassAtt) : -MAXFLOAT ),
		    ( HighT ? CVal(Case[Lp], ClassAtt) : MAXFLOAT ) );
    }

    /*  Carry out global check on remaining cases  */

    FindContinOutliers(Fp, Lp, true);
}



/*************************************************************************/
/*									 */
/*	Check continuous values of ClassAtt for cases Fp to Lp		 */
/*									 */
/*************************************************************************/


void FindContinOutliers(CaseNo Fp, CaseNo Lp, Boolean Sorted)
/*   ------------------  */
{
    CaseNo	Tail, Cases, LowTp=-1, HighTp=-1, GFp, i;
    double	Mean, SD, LowFrac, HighFrac, LowLim, HighLim;
    Clust	CLow=Nil, CHigh=Nil;
    Boolean	SavedCluster=false;

    Cases = No(Fp, Lp);
    if ( Cases < CMINITEMS ) return;

    if ( ! Sorted )
    {
	Quicksort(Fp, Lp, ClassAtt);
    }

    TrimmedSDEstimate(Fp, Lp, &Mean, &SD);

    /*  Check low and high tails.  A tail is anomalous if
	 * it does not contain too many cases
	 * the case before the tail has a Z-score <= MAXNORM
	 * there is a gap of at least MINABNORM - MAXNORM  */

    Tail = MaxAnoms(Cases);

    LowTp  = FindTail(Fp, Fp + Tail,  1, Mean, SD);
    HighTp = FindTail(Lp, Lp - Tail, -1, Mean, SD);

    if ( SIFT  )
    {
	/*  See whether we need to save this cluster for low or high test  */

	if ( LowTp >= 0 )
	{
	    LowFrac = No(LowTp+1, Lp) / (double) Cases;
	    LowLim  = CVal(Case[LowTp+1], ClassAtt);
	}
	else
	{
	    LowFrac = LowLim = 0;
	}

	if ( HighTp > 0 )
	{
	    HighFrac = No(Fp, HighTp-1) / (double) Cases;
	    HighLim  = CVal(Case[HighTp-1], ClassAtt);
	}
	else
	{
	    HighFrac = HighLim = 0;
	}

	if ( LowFrac > 0 || HighFrac > 0 )
	{
	    SaveContinCluster(Mean, SD, Cases,
			      LowFrac, LowLim, HighFrac, HighLim);

	    SavedCluster = true;
	}
    }


    if ( LowTp >= 0 || HighTp > 0 )
    {
	GFp = ( LowTp >= 0 ? LowTp+1 : Fp );

	if ( LowTp >= 0 )
	{
	    CLow = NewClust(Mean, SD,
			    CVal(Case[LowTp+1], ClassAtt),
			    No(Fp, LowTp), Cases);
	}

	if ( HighTp > 0 )
	{
	    CHigh = NewClust(Mean, SD,
			     CVal(Case[HighTp-1], ClassAtt),
			     No(HighTp, Lp), Cases);

	    /*  Move all anomalies to the front  */

	    ForEach(i, HighTp, Lp)
	    {
		Swap(i, GFp);
		GFp++;
	    }
	}

	LabelContinOutliers(CLow, CHigh, Fp, GFp, Lp);
    }

    /*  Clusters may have caveats discovered during LabelContinOutliers  */

    if ( SavedCluster )
    {
	ExtendSiftEntry("\n");
    }
}



/*************************************************************************/
/*									 */
/*	Note cases Fp to Lp as outliers wrt values GFp to GLp of	 */
/*	ClassAtt whose mean and SD are given				 */
/*									 */
/*************************************************************************/


void LabelContinOutliers(Clust CL, Clust CH, CaseNo Fp, CaseNo GFp, CaseNo GLp)
/*   -------------------  */
{
    CaseNo	i;
    double	Z, Mean, SD, X;
    Clust	C, OldC;

    C = ( CL ? CL : CH );	/* either will do since mean is the same */

    Mean = C->Expect;
    SD   = C->SD;

    /*  Remove cases that already have a more interesting recorded
	anomalous value  */

    ForEach(i, Fp, GFp-1)
    {
	/*  Use Chebychev bounds to approximate certainty that this
	    case is an outlier  */

	Z = ZScore(i);
	X = 1 / (Z * Z);

	if ( (OldC = OutClust(Case[i])) &&
	     ( C->NCond > OldC->NCond ||
	       C->NCond == OldC->NCond && X >= OutXVal(Case[i]) ) )
	{
	    Swap(i, Fp);
	    Fp++;
	}
    }

    /*  Remove possible anomalies that are non consistent with the
	ordinary cases  */

    Fp = NoOtherDifference(Fp, GFp-1, GFp, GLp);

    /*  Finally, record remaining cases  */

    ForEach(i, Fp, GFp-1)
    {
	Z = ZScore(i);
	Verbosity(1,
	    fprintf(Of, "****\tpotential outlier %g (%.1f sd) %s\n",
			CVal(Case[i], ClassAtt), Z,
			( LabelAtt ? SVal(Case[i], LabelAtt) : "" )))

	RecordOutlier(i, ( CClass(Case[i]) < Mean ? CL : CH ), 1 / (Z * Z));
    }
}



/*************************************************************************/
/*									 */
/*	Robust estimator of mean and SD.				 */
/*	Idea: exclude high/low tails of data, and adjust computed	 */
/*	mean and SD heuristically.					 */
/*	Note: unknown and N/A values must be removed and cases must be	 */
/*	sorted by ClassAtt before calling TrimmedSDEstimate.		 */
/*									 */
/*************************************************************************/


void TrimmedSDEstimate(CaseNo Fp, CaseNo Lp, double *Mean, double *SD)
/*   -----------------  */
{
    CaseNo	i, Tail, Cases, Quart;
    double	Val, Sum=0, SumSq=0;

    /*  Set defaults  */

    *Mean = 0;
    *SD   = 1E38;

    Tail = MaxAnoms(No(Fp, Lp));
    Cases = No(Fp, Lp) - 2 * Tail;
    Quart = No(Fp, Lp) / 4;

    /*  Don't try to estimate if too many values are the same  */

    if ( CClass(Case[Fp+Quart]) == CClass(Case[Lp-Quart]) )
    {
	return;
    }

    ForEach(i, Fp+Tail, Lp-Tail)
    {
	if ( NotApplic(Case[i], ClassAtt) )
	{
	    Cases--;
	}
	else
	{
	    Val    = CClass(Case[i]);
	    Sum   += Val;
	    SumSq += Val * Val;
	}
    }

    if ( Cases < Tail )
    {
	return;
    }

    /*  If there are N cases (excluding non-applicables) then adjust
	SD by factor (N + Tail) / (N - Tail)  */

    *Mean = Sum / Cases;
    *SD   = SDEstimate(Cases, Sum, SumSq) *
	    (Cases + 3.0 * Tail ) / (Cases + Tail);
}



/*************************************************************************/
/*									 */
/*	Find a tail containing potential anomalies between Fp and Lp-1.	 */
/*	* case Lp must have a Z-score <= MAXNORM			 */
/*	* there must be a gap >= MINABNORM-MAXNORM between the anomalous */
/*	  and non-anomalous values					 */
/*	* the cluster cannot contain cases from omitted tails		 */
/*									 */
/*************************************************************************/


CaseNo FindTail(CaseNo Fp, CaseNo Lp, int I, double Mean, double SD)
/*     --------  */
{
    CaseNo	i;
    double	Z;

    if ( ZScore(Lp) > MAXNORM ) return -1;

    /*  Find the first gap  */

    for ( i = Lp ; i * I > Fp * I && (Z = ZScore(i)) <= MINABNORM ; i -= I )
    {
	if ( ZScore(i - I) - Z >= MINABNORM - MAXNORM )
	{
	    break;
	}
    }

    return ( Z > MINABNORM ? -1 : OmittedCases(I) ? -1 : i - I );
}



/*************************************************************************/
/*									 */
/*	Check whether the current cluster includes cases from the	 */
/*	excluded high/low tails						 */
/*									 */
/*************************************************************************/


Boolean OmittedCases(int HiLo)
/*      ------------  */
{
    CaseNo	Fp, Lp;
    CaseNo	i;

    if ( HiLo > 0 )
    {
	Fp = LowFp;
	Lp = LowLp;
    }
    else
    {
	Fp = HighFp;
	Lp = MaxCase;
    }

    ForEach(i, Fp, Lp)
    {
	if ( SatisfiesTests(Case[i]) )
	{
	    return true;
	}
    }

    return false;
}



/*************************************************************************/
/*									 */
/*	See whether a case satisfies all current tests			 */
/*									 */
/*************************************************************************/


Boolean SatisfiesTests(Description Case)
/*      --------------  */
{
    Attribute	Att;
    DiscrValue	Br;
    int		i;

    ForEach(i, 0, GEnv.Level)
    {
	Att = GEnv.Test[i].Att;
	Br  = GEnv.Test[i].Br;

	if ( Unknown(Case, Att) )
	{
	    return false;
	}
	else
	if ( Br == 1 )
	{
	    if ( ! NotApplic(Case, Att) ) return false;
	}
	else
	if ( NotApplic(Case, Att) )
	{
	    return false;
	}
	else
	if ( Continuous(Att) )
	{
	    if ( ( Br == 2 ) != ( CVal(Case, Att) <= GEnv.Test[i].Cut ) )
	    {
		return false;
	    }
	}
	else
	if ( Ordered(Att) )
	{
	    if ( ( Br == 2 ) != ( DVal(Case, Att) <= GEnv.Test[i].Cut ) )
	    {
		return false;
	    }
	}
	else
	if ( Continuous(ClassAtt) && MaxAttVal[Att] > 3 )
	{
	    if ( ( Br == 2 ) != ( In(DVal(Case, Att), GEnv.Test[i].Left) != 0 ) )
	    {
		return false;
	    }
	}
	else
	if ( Br != DVal(Case, Att) )
	{
	    return false;
	}
    }

    return true;
}



/*************************************************************************/
/*									 */
/*	Check discrete values of ClassAtt for cases Fp to Lp		 */
/*	Idea: anomaly will appear as an odd case in nearly-pure subset	 */
/*									 */
/*************************************************************************/


void FindDiscrOutliers(CaseNo Fp, CaseNo Lp, CaseCount *Table)
/*   -----------------  */
{
    DiscrValue	v, Majority=1;
    CaseNo	i, GFp, GLp;
    CaseCount	Cases, Anoms;
    double	X;
    Clust	C, OldC;
    Boolean	SomeSurprise=false, NeedCluster;

    Cases = No(Fp, Lp);
    if ( Cases < DMINITEMS ) return;

    if ( ! Table )
    {
	FindClassFrequencies(Fp, Lp);
	Table = GEnv.ClassFreq;
    }

    ForEach(v, 2, MaxAttVal[ClassAtt])
    {
	if ( Table[v] > 0 )
	{
	    if ( ! Majority || Table[v] > Table[Majority] )
	    {
		Majority = v;
	    }
	}
    }

    /*  Skip if too many anomalies  */

    Anoms = Cases - Table[Majority];

    if ( Anoms > MaxAnoms(Cases) )
    {
	return;
    }

    /*  Check whether any non-majority class is surprising */

    for ( v = 1 ; ! SomeSurprise && v <= MaxAttVal[ClassAtt] ; v++ )
    {
	if ( v == Majority ) continue;

	X = XDScore(Table[v], Cases, Anoms, Prior[ClassAtt][v]);
	SomeSurprise = ( X <= 1.0 / (MINABNORM * MINABNORM) );
    }
    if ( ! SomeSurprise ) return;

    if ( SIFT )
    {
	SaveDiscrCluster(Majority, Anoms, Cases, Table);
    }

    if ( ! Anoms )
    {
	if ( SIFT )
	{
	    ExtendSiftEntry("\n");
	}

	return;
    }

    /*  Need a new cluster if surprising non-zero frequencies  */

    NeedCluster = ( Table[--v] > 0 );
    while ( ! NeedCluster && ++v <= MaxAttVal[ClassAtt] )
    {
	if ( v == Majority || ! Table[v] ) continue;

	X = XDScore(Table[v], Cases, Anoms, Prior[ClassAtt][v]);
	NeedCluster = ( X <= 1.0 / (MINABNORM * MINABNORM) );
    }

    if ( NeedCluster )
    {
	/*  Move all majority-class cases to the front  */

	GFp = Fp;
	GLp = (Fp = Group(ClassAtt, Majority, Fp, Lp, 0.0, Nil)) - 1;

	C = NewClust(Majority, 0.0, 0.0, Anoms, Cases);

	/*  Remove cases whose surprise value is insufficient or
	    that already have a more interesting recorded anomalous value  */

	ForEach(i, Fp, Lp)
	{
	    v = DClass(Case[i]);
	    X = DScore(Cases, Anoms, Prior[ClassAtt][v]);

	    if ( X > 1.0 / (MINABNORM * MINABNORM) ||
		 (OldC = OutClust(Case[i])) &&
		 ( C->NCond > OldC->NCond ||
		   C->NCond == OldC->NCond && X > OutXVal(Case[i]) ) )
	    {
		Swap(i, Fp);
		Fp++;
	    }
	}

	/*  Remove possible anomalies that are non consistent with the
	    ordinary cases  */

	Fp = NoOtherDifference(Fp, Lp, GFp, GLp);

	/*  Finally, record remaining cases  */

	if ( Fp <= Lp )
	{
	    ForEach(i, Fp, Lp)
	    {
		v = DClass(Case[i]);
		X = DScore(Cases, Anoms, Prior[ClassAtt][v]);

		Verbosity(1,
		    fprintf(Of, "****\tpotential outlier %s (p=%.3f) %s\n",
				AttValName[ClassAtt][DClass(Case[i])], X,
				( LabelAtt ? SVal(Case[i], LabelAtt) : "" )))

		RecordOutlier(i, C, X);
	    }
	}
    }

    if ( SIFT && SomeSurprise )
    {
	ExtendSiftEntry("\n");
    }
}



/*************************************************************************/
/*									 */
/*	Cases Fp through Lp have been identified as potential anoms	 */
/*	in the cluster whose "normal" cases are GFp thrrough GLp.	 */
/*	Discard potential anomalies that appear to be inconsistent	 */
/*	with the normals on some other attribute.			 */
/*	If SIFT is set, record any caveats for the current cluster	 */
/*									 */
/*************************************************************************/


CaseNo NoOtherDifference(CaseNo Fp, CaseNo Lp, CaseNo GFp, CaseNo GLp)
/*     -----------------  */
{
    Attribute	Att;
    double	Sum, SumSq, Mean, SD, CV;
    CaseNo	i, Cases, GCases;
    DiscrValue	v;
    Boolean	Caveat;
    int		Bytes;
    char	SE[100];

    if ( GEnv.Level < 0 ||
	 Fp > Lp || (GCases = No(GFp, GLp)) < MINCONTEXT ) return Fp;

    /*  Use a sample if there are many normal cases  */

    if ( GCases > MaxDiscrVal * SAMPLEUNIT )
    {
	GCases = 0.5 * MaxDiscrVal * SAMPLEUNIT;
	Sample(GFp, GLp, GCases);
	GLp = GFp + GCases - 1;
    }

    ForEach(Att, 1, MaxAtt)
    {
	if ( Att == ClassAtt || Exclude(Att) ) continue;

	if ( Fp > Lp ) return Fp;

	Caveat = false;

	if ( Continuous(Att) )
	{
	    /*  Find mean and variance of ordinary cases  */

	    Sum = SumSq = Cases = 0;
	    ForEach(i, GFp, GLp)
	    {
		if ( ! Unknown(Case[i], Att) && ! NotApplic(Case[i], Att) )
		{
		    CV = ( UseLogs[Att] ? log(CVal(Case[i], Att)) :
					  CVal(Case[i], Att) );
		    Sum   += CV;
		    SumSq += CV * CV;
		    Cases++;
		}
	    }

	    /*  Check that sufficient cases to give reliable SD  */

	    if ( Cases >= MINCONTEXT )
	    {
		Mean = Sum / Cases;
		SD   = SDEstimate(Cases, Sum, SumSq);

		/*  Move filtered cases to the front  */

		ForEach (i, Fp, Lp)
		{
		    if ( ! Unknown(Case[i], Att) &&
			 ! NotApplic(Case[i], Att) &&
			 fabs(Mean -
			      ( UseLogs[Att] ? log(CVal(Case[i], Att)) :
					       CVal(Case[i], Att) ))
			      / SD > MAXNORM )
		    {
			Verbosity(2,
			    fprintf(Of, "\t  %d: difference %s %.2f SD\n",
				    i, AttName[Att],
				    (Mean -
				     ( UseLogs[Att] ? log(CVal(Case[i], Att)) :
						  CVal(Case[i], Att) )) / SD))
			Swap(i, Fp);
			Fp++;

			/*  Record possible caveat  */

			if ( SIFT && ! Caveat )
			{
			    Caveat = true;

			    sprintf(SE, " %d", Att);
			    ExtendSiftEntry(SE);
			    if ( UseLogs[Att] )
			    {
				sprintf(SE, " %.8g %.8g",
					    exp(Mean - MAXNORM * SD),
					    exp(Mean + MAXNORM * SD));
			    }
			    else
			    {
				sprintf(SE, " %.8g %.8g",
					    Mean - MAXNORM * SD,
					    Mean + MAXNORM * SD);
			    }

			    ExtendSiftEntry(SE);
			}
		    }
		}
	    }
	}
	else
	{
	    /*  Discrete attribute
		NB: This doesn't differentiate between ordered and unordered
		    discrete attributes -- perhaps it should  */

	    ForEach(v, 0, MaxAttVal[Att])
	    {
		GEnv.ValFreq[v] = 0;
	    }

	    ForEach(i, GFp, GLp)
	    {
		GEnv.ValFreq[XDVal(Case[i], Att)]++;
	    }

	    /*  A discrete attribute value is judged to inconsistent with
		the normals if its Laplace probability in the normals is
		less than 0.025 and its prior greater than 0.25  */

	    Bytes = (MaxAttVal[Att]>>3) + 1;
	    ClearBits(Bytes, GEnv.Subset[0]);

	    ForEach(i, Fp, Lp)
	    {
		v = XDVal(Case[i], Att);
		if ( Prior[Att][v] >= 0.25 &&
		     (GEnv.ValFreq[v] + 1) / (double) (GCases + 2) < 0.025L )
		{
		    Verbosity(2,
			fprintf(Of, "\t  %d: difference %s=%s (%d/%d)\n",
				i, AttName[Att], AttValName[Att][v], 
				GEnv.ValFreq[v], GCases))
		    Swap(i, Fp);
		    Fp++;

		    SetBit(v, GEnv.Subset[0]);
		    Caveat = true;
		}
	    }

	    if ( SIFT && Caveat )
	    {
		sprintf(SE, " %d", Att);
		ExtendSiftEntry(SE);
		ForEach(v, 0, Bytes-1)
		{
		    sprintf(SE, " %x", GEnv.Subset[0][v]);
		    ExtendSiftEntry(SE);
		}
	    }
	}
    }

    return Fp;
}

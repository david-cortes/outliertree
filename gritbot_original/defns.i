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
/*		Definitions						 */
/*              -----------						 */
/*									 */
/*************************************************************************/


#define	 RELEASE	"2.01 GPL Edition"

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>
#include <limits.h>
#include <float.h>

#include "text.i"



/*************************************************************************/
/*									 */
/*		Definitions dependent on cc options			 */
/*									 */
/*************************************************************************/


#define	 Goodbye(x)		exit(x)
#define	 Of			stdout

#include <values.h>

#ifdef	VerbOpt
#include <assert.h>
#define	Verbosity(d,s)		if(VERBOSITY >= d) {s;}
#define	Free(x)			{free(x); x = 0;}
#else
#define	assert(x)
#define Verbosity(d,s)
#define	Free(x)			free(x)
#endif


/*************************************************************************/
/*									 */
/*		Constants, macros etc.					 */
/*									 */
/*************************************************************************/


#define	 MAXFRAC	0.01	/* max proportion of outliers in group */
#define	 MAXNORM	2.67	/* max SDs for non-outlier */
#define	 MAXTAIL	5.34	/* max SDs for 'ordinary' tail */
#define	 SAMPLEUNIT	2000	/* min sample per split */
#define	 SAMPLEFACTOR	5	/* threshold for using sampling */
#define	 MINCONTEXT	25	/* min cases to detect other difference */

#define	 Nil	   0		/* null pointer */
#define	 false	   0
#define	 true	   1
#define	 None	   -1
#define	 Epsilon   1E-6

#define  EXCLUDE   1		/* special attribute status: do not use */
#define	 SKIP	   2		/* do not check	*/
#define  DISCRETE  4		/* ditto: collect values as data read */
#define  ORDERED   8		/* ditto: ordered discrete values */
#define  DATEVAL   16		/* ditto: YYYY/MM/DD or YYYY-MM-DD */
#define  STIMEVAL  32		/* ditto: HH:MM:SS */
#define	 TSTMPVAL  64		/* date time */

				/* unknown and N/A values are represented by
				   unlikely floating-point numbers
				   (octal 01600000000 and 01) */
#define	 UNKNOWN   01600000000	/* 1.5777218104420236e-30 */
#define	 NA	   01		/* 1.4012984643248171e-45 */

#define	 BrDiscr   1
#define	 BrThresh  2
#define	 BrSubset  3

#define  AllocZero(N,T)		(T *) Pcalloc(N, sizeof(T))
#define  Alloc(N,T)		AllocZero(N,T) /*for safety */
#define  Realloc(V,N,T)		V = (T *) Prealloc(V, (N)*sizeof(T))

#define	 Max(a,b)               ((a)>(b) ? a : b)
#define	 Min(a,b)               ((a)<(b) ? a : b)

#define	 Log2			0.69314718055994530942

#define	 Bit(b)			(1 << (b))
#define	 In(b,s)		((s[(b) >> 3]) & Bit((b) & 07))
#define	 ClearBits(n,s)		memset(s,0,n)
#define	 CopyBits(n,f,t)	memcpy(t,f,n)
#define	 SetBit(b,s)		(s[(b) >> 3] |= Bit((b) & 07))
#define	 ResetBit(b,s)		(s[(b) >> 3] ^= Bit((b) & 07))

#define	 ForEach(v,f,l)		for(v=f ; v<=l ; ++v)

#define	 Swap(a,b)		{Description _xab; _xab=Case[a]; Case[a]=Case[b]; Case[b]=_xab;}

#define	 StatBit(a,b)		(SpecialStatus[a]&(b))
#define	 Exclude(a)		StatBit(a,EXCLUDE)
#define	 Skip(a)		StatBit(a,EXCLUDE|SKIP)
#define  Discrete(a)		(MaxAttVal[a] || StatBit(a,DISCRETE))
#define  Continuous(a)		(! MaxAttVal[a] && ! StatBit(a,DISCRETE))
#define	 Ordered(a)		StatBit(a,ORDERED)
#define	 DateVal(a)		StatBit(a,DATEVAL)
#define	 TimeVal(a)		StatBit(a,STIMEVAL)
#define	 TStampVal(a)		StatBit(a,TSTMPVAL)

#define  FreeUnlessNil(p)	if((p)!=Nil) free(p)

#define  CheckClose(f)		if(f) {fclose(f); f=Nil;}

#define  Space(s)	(s == ' ' || s == '\n' || s == '\r' || s == '\t')
#define  SkipComment	while ( (c = InChar(f)) != '\n' && c != EOF )

#define	 rint(x)	floor((x)+0.5)	/* for consistency across platforms */
#define	 P1(x)		(rint((x)*10) / 10)

#define	 No(f,l)	((l)-(f)+1)
#define  SplitVal(g,i)	( Continuous(ClassAtt) ? (g) : (g) / ((i) + 1E-6) )


#define	 NOFILE		 0
#define	 BADATTNAME	 1
#define	 EOFINATT	 2
#define	 SINGLEATTVAL	 3
#define	 BADATTVAL	 4
#define	 BADNUMBER	25
#define	 BADCLASS	 5
#define	 DUPATTNAME	 6
#define	 NOMEM		 8
#define	 TOOMANYVALS	 9
#define	 BADDISCRETE	10
#define	 UNKNOWNATT	11
#define	 LONGNAME	13
#define	 HITEOF		14
#define	 MISSNAME	15
#define	 BADDATE	16
#define	 BADTIME	17
#define	 BADDEF1	20
#define	 BADDEF2	21
#define	 BADDEF3	22
#define	 SAMEATT	23
#define	 BADTSTMP	24
#define	 BADSIFT	30

#define	 READDATA	1
#define	 READTEST	2
#define	 READCASES	3
#define	 PRELIM		4
#define	 CHECKING	5
#define	 REPORTING	6
#define	 CLEANUP	7

#define	 CONT_GT	1	/* CONT_GLT = CONT_GT | CONT_LT */
#define	 CONT_LT	2
#define	 CONT_GLT	3
#define	 CONT_NA	4
#define	 DISCR_GT	5	/* DISCR_GLT = DISCR_GT | DISCR_LT */
#define	 DISCR_LT	6
#define	 DISCR_GLT	7
#define	 DISCR_VAL	8
#define	 DISCR_SET	9


/*************************************************************************/
/*									 */
/*		Type definitions					 */
/*									 */
/*************************************************************************/


typedef  unsigned char	Boolean, BranchType, *Set;
typedef	 char		*String;

typedef  int	CaseNo,		/* data item number */
		CaseCount;	/* count of items */

typedef  int	DiscrValue,	/* discrete attribute value (0 = ?) */
		Attribute;	/* attribute number, 1..MaxAtt */


				/* defining USEDOUBLE allows for DP
				   attribute values, but will not affect
				   use of saved analyses */
#ifdef USEDOUBLE
typedef	 double	ContValue;	/* continuous attribute value */
#define	 PREC	14		/* precision */
#define	 MARKER	MAXDOUBLE
#else
typedef	 float	ContValue;	/* continuous attribute value */
#define	 PREC	 7		/* precision */
#define	 MARKER	MAXFLOAT
#endif


typedef  union	 _def_val
	 {
	    String	_s_val;		/* att val for comparison */
	    ContValue	_n_val;		/* number for arith */
	 }
	 DefVal;

typedef  struct  _def_elt
	 {
	    short	_op_code;	/* type of element */
	    DefVal	_operand;	/* string or numeric value */
	 }
	 DefElt, *Definition;

typedef  struct  _elt_rec
	 {
	    int		Fi,		/* index of first char of element */
			Li;		/* last ditto */
	    char	Type;		/* 'B', 'S', or 'N' */
	 }
	 EltRec;


#define	 DefOp(DE)	DE._op_code
#define	 DefSVal(DE)	DE._operand._s_val
#define	 DefNVal(DE)	DE._operand._n_val

#define	 OP_ATT			 0	/* opcodes */
#define	 OP_NUM			 1
#define	 OP_STR			 2
#define	 OP_MISS		 3
#define	 OP_AND			10
#define	 OP_OR			11
#define	 OP_EQ			20
#define	 OP_NE			21
#define	 OP_GT			22
#define	 OP_GE			23
#define	 OP_LT			24
#define	 OP_LE			25
#define	 OP_SEQ			26
#define	 OP_SNE			27
#define	 OP_PLUS		30
#define	 OP_MINUS		31
#define	 OP_UMINUS		32
#define	 OP_MULT		33
#define	 OP_DIV			34
#define	 OP_MOD			35
#define	 OP_POW			36
#define	 OP_SIN			40
#define	 OP_COS			41
#define	 OP_TAN			42
#define	 OP_LOG			43
#define	 OP_EXP			44
#define	 OP_INT			45
#define	 OP_END			99


typedef  struct _testrec
	 {
	    Attribute	Att;		/* attribute tested */
	    DiscrValue	Br;		/* branch of test */
	    ContValue	Cut;		/* threshold (if relevant) */
	    Set		Left;		/* values for left br (if relevant) */
	 }
	 TestRec;

typedef	 struct _clustcondrec
	 {
	    int		Type;		/* type of test */
	    Attribute	Att;		/* attribute tested */
	    ContValue	Low,		/* low thresh or start of range */
			High;		/* high threshold or end of range */
	    Set		Values;		/* value subset if required */
	 }
	 ClustCond;

typedef	 struct	_clustrec
	 {
	    Attribute	Att;		/* focus attribute */
	    ClustCond	*Cond;		/* group conditions */
	    int		NCond;		/* number of group conditions */
	    ContValue	Expect,		/* mean | (int) modal value */
			SD,		/* sd (trimmed) */
			Limit;		/* low / high value for normal cases */
	    float	Frac;		/* proportion of "normal" cases */
	    CaseCount	GpSize;		/* size of group */
	 }
	 ClustRec, *Clust;

typedef  union _attribute_value
	 {
	    DiscrValue	_discr_val;
	    ContValue	_cont_val;
	    String	_string_val;
	    Clust	_clust;
	 }
	 AttValue, *Description;

typedef  struct _sort_pair
         {
            ContValue   C;
            Description D;
         }
         SortPair;

typedef	 struct _caveat_rec
	 {
	    Attribute	Att;
	    Set		Subset;
	    float	Low, High;
	 }
	 CaveatRec;

typedef  struct _treerec	*Tree;
typedef  struct _treerec
	 {
	    BranchType	NodeType;	/* 0 | BrDiscr | BrThresh | BrSubset */
	    Attribute	Tested; 	/* attribute referenced in test */
	    int		Forks;		/* number of branches at this node */
	    ContValue	Cut;		/* threshold for continuous attribute */
	    Set         Left;		/* subset of values for first branch */
	    Tree	*Branch,	/* Branch[x] = subtree for outcome x */
			Parent;		/* parent node */
	    DiscrValue	Br;		/* branch from parent */
	    String	SiftEntry;	/* text for sift file */
	 }
	 TreeRec;

typedef	 struct _env_rec
	 {
	    CaseCount	**Freq,
			**BestFreq,
			*ValFreq,
			*ClassFreq,
			BrFreq[4];
	    Boolean	*Left,
			*Possible;
	    double	*Gain,
			*Info,
			*ValSum,
			*ValSumSq,
			BrSum[4],
			BrSumSq[4],
			BaseInfo,
			FRAC,
			PSD;
	    Set		*Subset;
	    ContValue	*Bar;
	    int		Level,
			MaxLevel,
			*Tested;
	    TestRec	*Test;
	    String	SiftEntry;
	    int		SiftSize,
			SiftSpace;
	    Attribute	*DList;		/* current discrete atts */
	    CaseCount	***DFreq;	/* DFreq[a][][] = Freq[] for att a */
	    double	**DValSum,	/* ValSum[] for att a */
			**DValSumSq;	/* ValSumSq[] for att a */
	 }
	 EnvRec;


#define  CVal(Case,Attribute)   Case[Attribute]._cont_val
#define  DVal(Case,Attribute)   Case[Attribute]._discr_val
#define	 XDVal(Case,Att)	(Case[Att]._discr_val & 077777777)
#define  SVal(Case,Attribute)   Case[Attribute]._string_val

#define  CClass(Case)		(*Case)._cont_val
#define  DClass(Case)		((*Case)._discr_val & 077777777)

#define	 Unknown(Case,Att)	(DVal(Case,Att)==UNKNOWN)
#define	 UnknownVal(AV)		(AV._discr_val==UNKNOWN)
#define	 NotApplic(Case,Att)	(DVal(Case,Att)==NA)
#define	 NotApplicVal(AV)	(AV._discr_val==NA)

#define	 OutXVal(Case)		Case[MaxAtt+1]._cont_val
#define	 OutClust(Case)		Case[MaxAtt+2]._clust

#define  ZScore(i)		(fabs(CClass(Case[i])-Mean) / SD)
#define	 MaxAnoms(N)		(MAXFRAC*(N)+2*sqrt((N)*MAXFRAC*(1-MAXFRAC))+1)

#define	 DScore(n,a,p)		((a) / ((double) (n)*(p)))

	/* XDScore is a specialised version for possibly non-occuring vals  */
#define	 XDScore(f,n,a,p)	((f) ? (a) / ((double) (n)*(p)) :\
				 (p) ? (1 / ((n)+2.0)) / (p) : (1 / ((n)+2.0)))


/*************************************************************************/
/*									 */
/*		Function prototypes					 */
/*									 */
/*************************************************************************/

	/* getnames.c */

Boolean	    ReadName(FILE *f, String s, int n, char ColonOpt);
void	    GetNames(FILE *Nf);
void	    ExplicitAtt(FILE *Nf);
int	    Which(String Val, String *List, int First, int Last);
String	    CopyString(String S);
void	    FreeNames(void);
int	    InChar(FILE *f);

	/* implicitatt.c */

void	    ImplicitAtt(FILE *Nf);
void	    ReadDefinition(FILE *f);
void	    Append(char c);
Boolean	    Expression(void);
Boolean	    Conjunct(void);
Boolean	    SExpression(void);
Boolean	    AExpression(void);
Boolean	    Term(void);
Boolean	    Factor(void);
Boolean	    Primary(void);
Boolean	    Atom(void);
Boolean	    Find(String S);
int	    FindOne(String *Alt);
Attribute   FindAttName(void);
void	    DefSyntaxError(String Msg);
void	    DefSemanticsError(int Fi, String Msg, int OpCode);
void	    Dump(char OpCode, ContValue F, String S, int Fi);
void	    DumpOp(char OpCode, int Fi);
Boolean	    UpdateTStack(char OpCode, ContValue F, String S, int Fi);
AttValue    EvaluateDef(Definition D, Description Case);

	/* getdata.c */

void	    GetData(FILE *Df, Boolean Train);
Description GetDescription(FILE *Df, Boolean Train);
void	    FreeData(void);
void	    FreeCase(Description DVec);
void	    CheckValue(Description DVec, Attribute Att);

	/* check.c */

void	    CheckData(void);
void	    CheckContin(CaseNo Fp);
void	    FindContinOutliers(CaseNo Fp, CaseNo Lp, Boolean Sorted);
void	    LabelContinOutliers(Clust CL, Clust CH, CaseNo Fp, CaseNo GFp,
				CaseNo GLp);
void	    TrimmedSDEstimate(CaseNo Fp, CaseNo Lp, double *Mean, double *SD);
CaseNo	    FindTail(CaseNo Fp, CaseNo Lp, int Inc, double Mean, double SD);
Boolean	    OmittedCases(int HiLo);
Boolean	    SatisfiesTests(Description Case);
void	    FindDiscrOutliers(CaseNo Fp, CaseNo Lp, CaseCount *Table);
CaseNo	    NoOtherDifference(CaseNo Fp, CaseNo Lp, CaseNo GFp, CaseNo GLp);
void	    InitialiseEnvData(void);
void	    FreeEnvData(void);

	/* cluster.c */

Clust	    NewClust(ContValue Expect, ContValue SD, ContValue Limit,
		     CaseCount Anoms, CaseCount GpSize);
void	    SaveClustConds(Clust C);
void	    FormatContinCond(Attribute Att, ClustCond *CC);
void	    FormatOrderedCond(Attribute Att, ClustCond *CC);
void	    FormatSubsetCond(Attribute Att, ClustCond *CC);
void	    FormatValCond(Attribute Att, ClustCond *CC);
void	    FreeClust(Clust C);

	/* outlier.c */

void	    RecordOutlier(CaseNo i, Clust C, float XVal);
void	    ReportOutliers(void);
void	    PrintAttVal(Description Case, Attribute Att);
void	    PrintOutlier(CaseNo i, Clust C, ContValue SVal);
void	    PrintContinCond(Attribute Att, ContValue Lo, ContValue Hi, CaseNo N);
void	    PrintOrderedCond(Attribute Att, DiscrValue Lo, DiscrValue Hi, CaseNo N);
void	    PrintSubsetCond(Attribute Att, Set Values, CaseNo N);
void	    PrintValCond(Attribute Att, DiscrValue v);


	/* common.c */

void	    InitialiseDAC(void);
void	    FreeDAC(void);
void	    Split(CaseNo Fp, CaseNo Lp, int CondAtts, Tree Parent,
		  DiscrValue Br, Tree *Result);
void	    RecoverContext(Tree T, DiscrValue Br);
void	    DiscreteAttInfo(CaseNo Fp, CaseNo Lp, int CondAtts);
void	    ChooseSplitWithSampling(CaseNo Fp, CaseNo Lp, int CondAtts);
void	    Sample(CaseNo Fp, CaseNo Lp, CaseCount N);
void	    SampleScan(CaseNo Fp, CaseNo Lp, int CondAtts, Boolean Second);
void	    ChooseSplit(CaseNo Fp, CaseNo Lp, int CondAtts);
void	    FindBestAtt(Attribute *BestAtt, double *BestVal);
void	    CheckSplit(Attribute Att, CaseNo Fp, CaseNo Lp);
void	    Divide(Tree Node, CaseNo Fp, CaseNo Lp, int CondAtts);
void	    NoteTest(Attribute Att, DiscrValue Br, ContValue Cut, Set Left);
CaseNo	    SkipMissing(Attribute Att, CaseNo Fp, CaseNo Lp);
CaseNo	    Group(Attribute Att, DiscrValue V, CaseNo Fp, CaseNo Lp,
		  ContValue Cut, Set Left);
void	    CheckPotentialClusters(Attribute Att, DiscrValue Forks,
				   CaseNo Fp, CaseNo Lp, ContValue B, Set S,
				   CaseCount **FT);
void	    ShowContext(CaseNo i);
Tree	    Leaf(Tree Parent, DiscrValue Br);
void	    ReleaseTree(Tree T, int Level);
void	    OutputConditions(void);

	/* continatt.c */

void	    CEvalContinAtt(Attribute Att, CaseNo Fp, CaseNo Lp);
ContValue   Between(ContValue Low, ContValue High);
void	    CEvalDiscrAtt(Attribute Att, CaseNo Fp, CaseNo Lp);
void	    EvalBinarySplit(Attribute Att, CaseNo Fp, CaseNo Lp);
void	    EvalSubsetSplit(Attribute Att, CaseNo Fp, CaseNo Lp);
double	    SDEstimate(CaseCount N, double Sum, double SumSq);
double	    ContinGain(void);

	/* discratt.c */

void	    DEvalContinAtt(Attribute Att, CaseNo Fp, CaseNo Lp);
void	    DEvalDiscrAtt(Attribute Att, CaseNo Fp, CaseNo Lp);
void	    DEvalOrderedAtt(Attribute Att, CaseNo Fp, CaseNo Lp);
void	    ComputeFrequencies(Attribute Att, CaseNo Fp, CaseNo Lp);
void	    FindClassFrequencies(CaseNo Fp, CaseNo Lp);
double	    DiscrGain(DiscrValue MaxVal, CaseCount TotalCases);
double	    TotalInfo(CaseCount V[], DiscrValue MinVal, DiscrValue MaxVal);

	/* sort.c */

void	    Quicksort(CaseNo Fp, CaseNo Lp, Attribute Att);
void	    Cachesort(CaseNo Fp, CaseNo Lp);

	/* modelfiles.c */

void	    CheckFile(String Extension, Boolean Write);
void	    WriteFilePrefix(String Extension);
void	    SaveCondition(void);
void	    SaveDiscrCluster(DiscrValue Expect, CaseCount Anoms,
			     CaseCount Cases, CaseCount *Freq);
void	    SaveContinCluster(float Mean, float SD, CaseCount Cases,
			      float LFrac, float LLim, float HFrac, float HLim);
void	    SaveNames(void);
void	    AsciiOut(String Pre, String S);
void	    ReadFilePrefix(String Extension);
int	    ReadProp(char *Delim);
String	    RemoveQuotes(String S);
void	    ExtendSiftEntry(String S);
void	    ProcessSift(void);
void	    Case1(void);
void	    Case2(void);
void	    Case3(void);
void	    Case11(void);
void	    Case12(void);
void	    Case13(void);
void	    Case21(void);
void	    Case22(void);
void	    ReadCaveats(void);
Boolean	    CheckCaveats(Description Case);
void	    FoundPossibleAnom(CaseNo i, Clust C, float Xv);
void	    Filter(Attribute Att, DiscrValue Br, ContValue Cut, Set Left);
void	    SetTestedAtts(void);


	/* utility.c */

void	    PrintHeader(String Title);
char	    ProcessOption(int Argc, char **Argv, char *Str);
void	    *Pmalloc(size_t Bytes);
void	    *Prealloc(void *Present, size_t Bytes);
void	    *Pcalloc(size_t Number, unsigned int Size);
void	    FreeVector(void **V, int First, int Last);
Description NewCase(void);
void	    MemTrim(void);
void	    FreeCases(void);
void	    FreeLastCase(Description Case);
double	    KRandom(void);
void	    ResetKR(int KRInit);
void	    Error(int ErrNo, String S1, String S2);
String	    CaseLabel(CaseNo N);
FILE *	    GetFile(String Extension, String RW);
double	    ExecTime(void);
int         Denominator(ContValue Val);
int         FracBase(Attribute Att);
int	    GetInt(String S, int N);
int	    DateToDay(String DS);
void	    DayToDate(int DI, String Date);
int	    TimeToSecs(String TS);
void	    SecsToTime(int Secs, String Time);
void	    SetTSBase(int y);
int	    TStampToMins(String TS);
void	    CValToStr(ContValue CV, Attribute Att, String DS);
void	    CleanupSift(void);
void	    Cleanup(void);
void	    Check(float Val, float Low, float High);


	/* update.c */

void	    NotifyStage(int);
void	    Progress(int);

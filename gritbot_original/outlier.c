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
/*	Routines for recording, reporting, saving and recovering	 */
/*	possible outliers						 */
/*	--------------------------------------------------------	 */
/*									 */
/*************************************************************************/


#include "defns.i"
#include "extern.i"


/*************************************************************************/
/*									 */
/*	Record outlier information for a case in cluster C		 */
/*									 */
/*************************************************************************/


void RecordOutlier(CaseNo i, Clust C, float XVal)
/*   -------------  */
{
    OutXVal(Case[i])  = XVal;
    OutClust(Case[i]) = C;
}



/*************************************************************************/
/*									 */
/*	Print outlier reports						 */
/*									 */
/*************************************************************************/


void ReportOutliers()
/*   --------------  */
{
    CaseNo	i, j, *Show, NShow=0, Stop=0;
    Boolean	FirstFromTest=true;
    FILE	*Lf;

    Show = Alloc(MaxCase+1, CaseNo);

    NotifyStage(REPORTING);
    Progress(-1);

    /*  Isolate outlier items  */

    if ( LIST && ! (Lf = GetFile(".list", "w")) )
    {
	Error(NOFILE, "", " for writing");
    }

    ForEach(i, 0, MaxCase)
    {
	if ( OutClust(Case[i]) )
	{
	    Show[NShow++] = i;

	    if ( LIST )
	    {
		if ( i > LastDataCase && FirstFromTest )
		{
		    fprintf(Lf, "\n");
		    FirstFromTest = false;
		}

		fprintf(Lf, "%d\n",
			    ( i <= LastDataCase ? i+1 : i - LastDataCase ));
	    }
	}
    }

    if ( LIST ) fclose(Lf);

    /*  Print outliers in descending order of confidence.  If MAXOUT
	is set, show only the first MAXOUT  */

    fprintf(Of, F_PossAnomalies(NShow));

    if ( MAXOUT > 0 && NShow > MAXOUT )
    {
	Stop = NShow - MAXOUT;
    }

    while ( NShow > Stop )
    {
	j = 0;
	for ( i = 1 ; i < NShow ; i++ )
	{
	    if ( OutXVal(Case[Show[i]]) < OutXVal(Case[Show[j]]) ||
		 OutXVal(Case[Show[i]]) == OutXVal(Case[Show[j]]) &&
		 Show[i] < Show[j] )
	    {
		j = i;
	    }
	}

	PrintOutlier(Show[j], OutClust(Case[Show[j]]), OutXVal(Case[Show[j]]));
	Show[j] = Show[--NShow];
    }

    Free(Show);
}



/*************************************************************************/
/*									 */
/*	Print the anomalous value and its context, then the		 */
/*	conditions that define the subset				 */
/*									 */
/*************************************************************************/


void PrintOutlier(CaseNo i, Clust C, ContValue XVal)
/*   ------------  */
{
    char	CVS1[20], CVS2[20];
    int		d;
    Attribute	Att;
    float	Mean;
    double	Base;
	
    /*  Identify the case  */

    if ( i > LastDataCase )
    {
	fprintf(Of, F_NoTestCase(i - LastDataCase));
    }
    else
    if ( LastDataCase < MaxCase )
    {
	fprintf(Of, F_NoDataCase(i+1));
    }
    else
    {
	fprintf(Of, F_NoCase(i+1));
    }
    if ( LabelAtt && SVal(Case[i], LabelAtt) )
    {
	fprintf(Of, F_LabelCase(CaseLabel(i)));
    }
    fprintf(Of, "  [%.3f]\n", XVal);

    /*  Show the primary attribute whose value is suspect  */

    fprintf(Of, "\t");
    PrintAttVal(Case[i], C->Att);
    fprintf(Of, F_Cases(C->GpSize));
    if ( Continuous(C->Att) )
    {
	Mean = ( UseLogs[C->Att] ? exp(C->Expect) : C->Expect );
	Base = pow(10.0, Prec[C->Att]);
	CValToStr(rint(Mean * Base) / Base, C->Att, CVS1);
	CValToStr(C->Limit,  C->Att, CVS2);
	fprintf(Of, F_CvGroup(CVS1,
		    ( C->GpSize < 100 ? 0 : C->GpSize < 1000 ? 1 : 2 ),
		    C->Frac * 100,
		    ( Mean < CVal(Case[i], C->Att) ? "<=" : ">=" ),
		    CVS2));
    }
    else
    {
	fprintf(Of, F_DvGroup(
		    ( C->GpSize < 100 ? 0 : C->GpSize < 1000 ? 1 : 2 ),
		    C->Frac * 100,
		    AttValName[C->Att][(int) C->Expect]));
    }

    /*  Show any conditioning tests  */

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
	}
	else
	{
	    PrintValCond(Att, (int) C->Cond[d].Low);
	}
    }
}



/*************************************************************************/
/*									 */
/*	Print an attribute value					 */
/*									 */
/*************************************************************************/


void PrintAttVal(Description Case, Attribute Att)
/*   -----------  */
{
    char	CVS[20];

    fprintf(Of, "%s = ", AttName[Att]);

    if ( Unknown(Case, Att) )
    {
	fprintf(Of, "?");
    }
    if ( NotApplic(Case, Att) )
    {
	fprintf(Of, "N/A");
    }
    else
    if ( Continuous(Att) )
    {
	CValToStr(CVal(Case, Att), Att, CVS);
	fprintf(Of, "%s", CVS);
    }
    else
    {
	fprintf(Of, "%s", AttValName[Att][XDVal(Case, Att)]);
    }
}



/*************************************************************************/
/*									 */
/*	Print a condition defining a subset (cluster).			 */
/*	Different functions are called for different formats etc.	 */
/*									 */
/*************************************************************************/


void PrintContinCond(Attribute Att, ContValue Lo, ContValue Hi, CaseNo N)
/*   ---------------  */
{
    char	CVS1[20], CVS2[20];

    fprintf(Of, "\t    %s ", AttName[Att]);

    if ( Lo > Hi )
    {
	fprintf(Of, "= N/A\n");
    }
    else
    {
	if ( Lo <= -MARKER )
	{
	    CValToStr(Hi, Att, CVS1);
	    fprintf(Of, "<= %s", CVS1);
	}
	else
	if ( Hi >= MARKER )
	{
	    CValToStr(Lo, Att, CVS1);
	    fprintf(Of, "> %s", CVS1);
	}
	else
	{
	    CValToStr(Lo, Att, CVS1);
	    CValToStr(Hi, Att, CVS2);
	    fprintf(Of, "> %s " T_and " <= %s", CVS1, CVS2);
	}

	CValToStr(CVal(Case[N], Att), Att, CVS1);
	fprintf(Of, " [%s]\n", CVS1);
    }
}



void PrintOrderedCond(Attribute Att, DiscrValue Lo, DiscrValue Hi, CaseNo N)
/*   ----------------  */
{
    fprintf(Of, "\t    %s ", AttName[Att]);

    if ( Lo == 1 && Hi == 1 )
    {
	fprintf(Of, "= N/A\n");
    }
    else
    if ( Lo == Hi )
    {
	fprintf(Of, "= %s\n", AttValName[Att][Lo]);
    }
    else
    {
	fprintf(Of, T_in " %s .. %s [%s]\n",
		    AttValName[Att][Lo], AttValName[Att][Hi],
		    AttValName[Att][DVal(Case[N], Att)]);
    }
}



void PrintSubsetCond(Attribute Att, Set Values, CaseNo N)
/*   ---------------  */
{
    DiscrValue	v, Last;
    int		Elts=0, Col, Entry;
    Boolean	First=true;

    if ( In(1, Values) )
    {
	fprintf(Of, "\t    %s = N/A\n", AttName[Att]);
	return;
    }

    /*  Special case for singleton  */

    ForEach(v, 2, MaxAttVal[Att])
    {
	if ( In(v, Values) )
	{
	    Elts++;
	    Last = v;
	}
    }

    if ( Elts == 1 )
    {
	fprintf(Of, "\t    %s = %s\n",
		    AttName[Att], AttValName[Att][Last]);
	return;
    }

    /*  Print the subset, breaking lines appropriately  */

    fprintf(Of, "\t    %s " T_in " {", AttName[Att]);
    Col = strlen(AttName[Att]) + 9;

    ForEach(v, 2, MaxAttVal[Att])
    {
	if ( In(v, Values) )
	{
	    if ( First )
	    {
		First = false;
	    }
	    else
	    {
		fprintf(Of, ", ");
		Col += 2;
	    }

	    Entry = strlen(AttValName[Att][v]);
	    if ( Col + Entry >= 70 )
	    {
		Col = strlen(AttName[Att]) + 9;
		fprintf(Of, "\n\t%*s", Col, "");
	    }
	    fprintf(Of, "%s", AttValName[Att][v]);
	    Col += Entry;
	}
    }
    fprintf(Of, "}");
    Col++;
    
    /*  Now print the actual value  */

    v = DVal(Case[N], Att);
    if ( Col + strlen(AttValName[Att][v]) + 3 > 72 )
    {
	fprintf(Of, "\n\t   ");
    }
    fprintf(Of, " [%s]\n", AttValName[Att][v]);
}



void PrintValCond(Attribute Att, DiscrValue v)
/*   ------------  */
{
    fprintf(Of, "\t    %s = %s\n",
		AttName[Att], AttValName[Att][v]);
}

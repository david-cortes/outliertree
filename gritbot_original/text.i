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
/*	Text strings for UTF-8 internationalization			 */
/*	-------------------------------------------			 */
/*									 */
/*************************************************************************/


	/*  General stuff  */


#ifdef UTF8
#define	 CharWidth(S)		UTF8CharWidth(S)
#else
#define	 CharWidth(S)		(int) strlen(S)
#endif

	/*  Strings etc  */

#define	 T_GritBot		"GritBot"
#define	 F_Release(n)		"Release " n

#define	 T_Options		"Options"
#define	 F_Application		"\tApplication `%s'\n"
#define	 F_Filtering		"\tFiltering level %g%%\n"
#define	 F_MaxConds		"\tMaxium %d conditions in description\n"
#define	 F_MinSubset		"\tMinimum %d cases in subset\n"
#define	 F_MaxOut		"\tShow up to %d possible anomalies\n"
#define	 F_NoSift		"\tDo not save analysis in .sift file\n"
#define	 F_ListAnoms		"\tSave case numbers of possible anomalies\n"
#define	 F_UnrecogOpt		"\n    **  Unrecognised option %s"\
				"\n    **  Summary of options for GritBot:\n"
#define	 F_OptList		"\t-f <filestem>\tapplication filestem\n"\
				"\t-l <percent>\tfiltering level\n"\
				"\t-c <integer>\tmaximum conditions in"\
					" description\n"\
				"\t-n <cases>\tlimit number of possible"\
					" anomalies reported\n"\
				"\t-s\t\tdo not save .sift file\n"\
				"\t-r\t\trecord case numbers of"\
					" possible anomalies\n"\
				"\t-h\t\tprint this message\n"
#define	 F_ReadData(c,a,f)	"\nRead %d cases (%d attributes) from"\
					" %s.data\n", c, a, f
#define	 F_ReadTest(c,f)	"Read %d cases from %s.test\n", c, f
#define	 F_AttChecked		"\nAttributes checked:\n"
#define	 F_AttNotChecked	"\nAttributes not checked:\n"
#define	 F_Time(s)		"\n\nTime: %.1f secs\n", s

#define	 F_WhileCheck		"\n    while checking %s:\n"
#define	 F_ExcludeMissing(a)	( (a) > 1 ?\
				  "\texcluding %d missing values\n" :\
				  "\texcluding %d missing value\n" ), a
#define	 F_ExcludeNA(a)		( (a) > 1 ?\
				  "\texcluding %d N/A values\n" :\
				  "\texcluding %d N/A value\n" ), a
#define	 F_TooManyIdentical	"\ttoo many identical values -- excluded\n"
#define	 F_LowTail(c,t)		"\texcluding low tail (%d cases below %s)\n",\
					c, t
#define	 F_HighTail(c,t)	"\texcluding high tail (%d cases above %s)\n",\
					c, t

#define	 F_WarnDemo		"\n\t** This demonstration version cannot"\
					" process **\n"\
				"\t** more than %d training or test cases."\
					"     **\n"\

#define	 F_PossAnomalies(a)	( (a) != 1 ?\
				  "\n%d possible anomalies identified\n" :\
				  "\n%d possible anomaly identified\n" ), a
#define	 F_NoTestCase(c)	"\ntest case %d:", c
#define	 F_NoDataCase(c)	"\ndata case %d:", c
#define	 F_NoCase(c)		"\ncase %d:", c
#define	 F_LabelCase(l)		" (label %s)", l
#define	 F_Cases(c)		"  (%d cases, ", c
#define	 F_CvGroup(m,d,p,r,v)	"mean %s, %.*f%% %s %s)\n", m, d, p, r, v
#define	 F_DvGroup(d,p,v)	"%.*f%% `%s')\n", d, p, v
#define	 T_and			"and"
#define	 T_in			"in"		/* element of set */

#define	 T_ReadTrain		"Reading training data"
#define	 T_ReadTest		"Reading test data"
#define	 T_Prelim		"Preliminaries"
#define	 T_Checking		"Checking"
#define	 T_Reporting		"Reporting anomalies"
#define	 T_CleaningUp		"Cleaning up"
#define	 F_Preliminaries	"Preliminaries for %-21.21s\n"
#define	 F_Checking(a,x,y,c)	"Checking %-21.21s  %s%s  (%d cases"\
					" checked)\n", a, x, y, c

#define	 F_Line(l,f)		"*** line %d of `%s': ", l, f
#define	 E_NOFILE(f,e)		"cannot open file %s%s\n", f, e
#define	 E_ForWrite		" for writing"
#define	 E_BADATTNAME		"`:' or `:=' expected after attribute name"\
					" `%s'\n"
#define	 E_UNKNOWNATT		"unknown attribute name `%s'\n"
#define	 E_EOFINATT		"unexpected eof while reading attribute `%s'\n"
#define	 E_SINGLEATTVAL(a,v)	"attribute `%s' has only one value `%s'\n",\
					a, v
#define	 E_DUPATTNAME		"multiple attributes with name `%s'\n"
#define	 E_BADATTVAL(v,a)	"bad value of `%s' for attribute `%s'\n", v, a
#define	 E_BADNUMBER(a)		"value of `%s' changed to `?'\n", a
#define	 E_BADCLASS		"bad class value `%s'l\n"
#define	 E_NOMEM		"unable to allocate sufficient memory\n"
#define	 E_TOOMANYVALS(a,n)	"too many values for attribute `%s'"\
					" (max %d)\n", a, n
#define	 E_BADDISCRETE		"bad number of discrete values for attribute"\
					" `%s'\n"
#define	 E_LONGNAME		"overlength name: check data file formats\n"
#define	 E_HITEOF		"unexpected end of file\n"
#define	 E_MISSNAME		"missing name or value before `%s'\n"
#define	 E_BADTSTMP(d,a)	"bad timestamp `%s' for attribute `%s'\n", d, a
#define	 E_BADDATE(d,a)		"bad date `%s' for attribute `%s'\n", d, a
#define	 E_BADTIME(d,a)		"bad time `%s' for attribute `%s'\n", d, a
#define	 E_BADDEF1(a,s,x)	"in definition of attribute `%s':\n"\
					"\tat `%.12s': expect %s\n", a, s, x
#define	 E_BADDEF2(a,s,x)	"in definition of attribute `%s':\n"\
					"\t`%s': %s\n", a, s, x
#define	 E_SAMEATT(a,b)		"attribute `%s' is identical to attribute"\
					" `%s'\n", a, b
#define	 E_BADDEF3		"cannot define target attribute `%s'\n"
#define	 E_SIFT			"sift file corrupted (entry \"%s\")\n"
#define	 T_ErrorLimit		"\nError limit exceeded\n"

#define	 F_CkOptList		"\t-f <filestem>\tapplication filestem\n"\
				"\t-n <cases>\tlimit number of possible"\
					" anomalies reported\n"\
				"\t-r\t\trecord case numbers of"\
					" possible anomalies\n"\
				"\t-h\t\tprint this message\n"
#define	 F_ReadSift		"\nRead saved analysis from %s.sift\n"
#define	 F_ReadCases(c,a,f)	"Read %d cases (%d attributes) from"\
					" %s.cases\n", c, a, f

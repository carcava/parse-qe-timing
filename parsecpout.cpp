#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<list>

using namespace std;

const int debug = 0;

const int NKEY = 33;
const int KMAXLEN = 32;

double get_wtime(const char * line);

/*
 */

enum {
	initialize,
	main_loop,
	cpr_total,
	cpr_md,
	move_electro,
	rhoofr,
	vofrho,
	dforce,
	calphi,
	newd,
	ortho_iter,
	rsg,
	rhoset,
	sigset,
	tauset,
	ortho,
	updatc,
	prefor,
	nlsm1,
	fft,
	ffts,
	fft_scatter,
	betagx,
	qradx,
	gram,
	nlinit,
	init_dim,
	newnlinit,
	from_scratch,
	strucf,
	calbec,
	ALLTOALL,
	CP
};

/*
 

*/

char keyword[NKEY][KMAXLEN] = {
"     initialize   :",
"     main_loop    :",
"     cpr_total    :",
"     cpr_md       :",
"     move_electro :",
"     rhoofr       :",
"     vofrho       :",
"     dforce       :",
"     calphi       :",
"     newd         :",
"     ortho_iter   :",
"     rsg          :",
"     rhoset       :",
"     sigset       :",
"     tauset       :",
"     ortho        :",
"     updatc       :",
"     prefor       :",
"     nlsm1        :",
"     fft          :",
"     ffts         :",
"     fft_scatter  :",
"     betagx       :",
"     qradx        :",
"     gram         :",
"     nlinit       :",
"     init_dim     :",
"     newnlinit    :",
"     from_scratch :",
"     strucf       :",
"     calbec       :",
"     ALLTOALL     :",
"     CP           :" };

char clean_keyword[NKEY][KMAXLEN];




char * CleanKey(const char * keyword) {
	static char cleankey[KMAXLEN];
	const char * pkey = keyword;
	while (*pkey == ' ') pkey++;
	strcpy(cleankey, pkey);
	int i = strlen(cleankey)-1;
	while (i >= 0 && (cleankey[i] == ' ' || cleankey[i] == ':')) i--;
	if (i < strlen(cleankey)) cleankey[i + 1] = '\0';
	return cleankey;
}

char * CleanDate(const char * date ) {
	static char cleandate[KMAXLEN];
	while (*date == ' ') date++;
	char day[4];
	day[0] = *date++;
	if( *date >= '0' && *date <= '9' ) {
		day[1] = *date++;
	} else {
		day[1] = day[0];
		day[0] = '0';
		day[2] = '\0';
	}

	int j = 0;
	cleandate[j++] = day[0];
	cleandate[j++] = day[1];
        // month
	cleandate[j++] = *date++;
	cleandate[j++] = *date++;
	cleandate[j++] = *date++;
	// year
	cleandate[j++] = *date++;
	cleandate[j++] = *date++;
	cleandate[j++] = *date++;
	cleandate[j++] = *date++;
	while ( *date < '0' || *date > '9' ) date++;
	cleandate[j++] = '@';

	char h[4];
	h[0] = *date++;
	if( *date >= '0' && *date <= '9' ) {
		h[1] = *date++;
	} else {
		h[1] = h[0];
		h[0] = '0';
		h[2] = '\0';
	}
	cleandate[j++] = h[0];
	cleandate[j++] = h[1];

	cleandate[j++] = *date++;
	
	while (*date == ' ') date++;
	char m[4];
	m[0] = *date++;
	if( *date >= '0' && *date <= '9' ) {
		m[1] = *date++;
	} else {
		m[1] = m[0];
		m[0] = '0';
		m[2] = '\0';
	}
	cleandate[j++] = m[0];
	cleandate[j++] = m[1];

	cleandate[j++] = *date++;

	while (*date == ' ') date++;
	char s[4];
	s[0] = *date++;
	if( *date >= '0' && *date <= '9' ) {
		s[1] = *date++;
	} else {
		s[1] = s[0];
		s[0] = '0';
		s[2] = '\0';
	}
	cleandate[j++] = s[0];
	cleandate[j++] = s[1];

	

	cleandate[j] = '\0';
	return cleandate;
}

/*
	 Parallel version (MPI & OpenMP), running on     288 processor cores
	 Number of MPI processes:               144
	 Threads/MPI process:                     2
	 K-points division:     npool     =       2
	 R & G space division:  proc/nbgrp/npool/nimage =      72
	 Reading input from scf_Fe_GR8x8.in

	 Current dimensions of program PWSCF are:
	 Max number of different atomic species (ntypx) = 10
	 Max number of k-points (npk) =  40000
	 Max angular momentum in pseudopotentials (lmaxx) =  3
			   file C.pbe-n-kjpaw_psl.0.1.UPF: wavefunction(s)  2P renormalized
			   file N.pbe-n-kjpaw_psl.0.1.UPF: wavefunction(s)  2P renormalized
			   file Fe.pbe-spn-kjpaw_psl.0.2.1.UPF: wavefunction(s)  3D renormalized

	 gamma-point specific algorithms are used

	 Subspace diagonalization in iterative solution of the eigenvalue problem:
	 one sub-group per band group will be used
	 ELPA distributed-memory algorithm (size of sub-group:  6*  6 procs)

 */

const char ncore_str[] = "Parallel version (MPI & OpenMP), running on";
const char mpitask_str[] = "Number of MPI processes:";
const char ompthread_str[] = "Threads/MPI process:";
const char npool_str[] = "K-points division:     npool     =";
const char ntg_str[] = "wavefunctions fft division:  fft and procs/group =";
const char northo_str[] = "ortho sub-group = ";
const char nbgrp_str[] = "band groups division:  nbgrp     =";
const char qever_str[] = "Program CP v.";

class ParaGeom {
	int ncore;
	int mpitask;
	int ompthread;
	int npool;
	int ntg;
	int northo;
	int nbgrp;
	char rev[8];
	char qever[4];
	char date[32];
        int ore, min, sec;
public:
	ParaGeom() {
		ncore = mpitask = ompthread = npool = ntg = northo = nbgrp = 1;
		qever[0] = '\0';
		rev[0]='\0';
		date[0]='\0';
	}

	void ParseRev( const char * a ) {
		while( strncmp(a++,"svn rev. ",9) && *a && ( *a != '\n' ) && ( *a != '\r' ) );
		a += 9;
		while (*a == ' ') a++; 
		rev[0] = *a; a++;
		rev[1] = *a; a++;
		rev[2] = *a; a++;
		rev[3] = *a; a++;
		rev[4] = '\0';
	}
	void ParseDate( const char * a ) {
		while( strncmp(a++,"starts on ",10) );
		a += 10;
		while (*a == ' ') a++; 
		for(int j=0;j<21;j++) date[j] = *a++;
		date[21] = '\0';
	}

	void Parse(const char * a) {
		while (*a == ' ') a++; // get rid of leading spaces
		if (!strncmp(a, ncore_str, strlen(ncore_str))) {
			if (debug) printf("%s\n", a);
			a = a + strlen(ncore_str);
			while (*a == ' ') a++; // get rid of leading spaces
			ncore = atoi(a);
		}
		if (!strncmp(a, mpitask_str, strlen(mpitask_str))) {
			if (debug) printf("%s\n", a);
			a = a + strlen(mpitask_str);
			while (*a == ' ') a++; // get rid of leading spaces
			mpitask = atoi(a);
			ompthread = ncore / mpitask;
		}
		if (!strncmp(a, npool_str, strlen(npool_str))) {
			if (debug) printf("%s\n", a);
			a = a + strlen(npool_str);
			while (*a == ' ') a++; // get rid of leading spaces
			npool = atoi(a);
		}
		if (!strncmp(a, ntg_str, strlen(ntg_str))) {
			if (debug) printf("%s\n", a);
			a = a + strlen(ntg_str);
			while (*a == ' ') a++; // get rid of leading spaces
			ntg = atoi(a);
		}
		if (!strncmp(a, northo_str, strlen(northo_str))) {
			if (debug) printf("%s\n", a);
			a = a + strlen(northo_str);
			while (*a == ' ') a++; // get rid of leading spaces
			northo = atoi(a);
			northo *= northo;
		}
		if (!strncmp(a, nbgrp_str, strlen(nbgrp_str))) {
			if (debug) printf("%s\n", a);
			a = a + strlen(nbgrp_str);
			while (*a == ' ') a++; // get rid of leading spaces
			nbgrp = atoi(a);
		}
		if (!strncmp(a, qever_str, strlen(qever_str))) {
			if (debug) printf("%s\n", a);
			qever[0] = a[18];
			qever[1] = a[20];
			qever[3] = '\0';
			ParseRev( a );
			ParseDate( a );
		}

	}
	int GetCores() { return ncore; };
	int GetTasks() { return mpitask; };
	int GetThreads() { return ompthread; };
	int GetPools() { return npool; };
	int GetTG() { return ntg; };
	int GetOrtho() { return northo; };
	int GetNbgrp() { return nbgrp; };
	const char * GetDate() const { return date; };
	void CheckQEver() { 
		if (qever[0]) {
			if (debug) printf("QE version tags: %c . %c\n",qever[0],qever[1]);
		}
		else {
			printf("WARNING: QE version tag was not found\n");
		}
	};
	void PrintHead(FILE * fp, bool cr = false) {
		fprintf(stdout, "Cores  MPI    OMP    Pools  TG     Ndiag  NBgrp  Date                  ");
		if (cr) fprintf(fp, "\n");
	}
	void PrintVal(FILE * fp, bool cr = false) {
		fprintf(stdout, "%-6d %-6d %-6d %-6d %-6d %-6d %-6d %-21s ", GetCores(), GetTasks(), GetThreads(), GetPools(), GetTG(), GetOrtho(), GetNbgrp(), CleanDate(GetDate()));
		if (cr) fprintf(fp, "\n");
	}
	int Print(FILE * fp, bool header = true) {
		if (header)
			PrintHead(fp, true);
		PrintVal(fp, true);
		return 0;
	}
};


class Timing {
	double t[NKEY];
	int tlen[NKEY];
public:
	Timing() {
		for (int i = 0; i < NKEY; i++) t[i] = 0.0;
		for (int i = 0; i < NKEY; i++) tlen[i] = strlen(keyword[i]);
	}
	int GetKey( const char * line ) {
		for (int j = 0; j < NKEY; j++) {
			if (!strncmp(line, keyword[j], tlen[j])) {
				if (debug) printf("%s", line);
				t[j] = get_wtime(line);
			}
		}
		return 0;
	}
	void PrintHead(FILE * fp, list<const char *> & keyout, bool cr = false) {
		for (int j = 0; j < NKEY; j++) {
			if (keyout.size() > 0) {
				bool ok = false;
				for (list<const char *>::const_iterator i = keyout.begin(); i != keyout.end(); i++) {
					if (!strcmp(*i, clean_keyword[j]) ) ok = true;
				}	
				if( ok ) 
					fprintf(fp, "%-14s", clean_keyword[j]);
			}
			else {
				fprintf(fp, "%-14s", clean_keyword[j]);
			}
		}
		if( cr ) fprintf(fp, "\n");
	}
	void PrintVal(FILE * fp, list<const char *> & keyout, bool cr = false) {
		for (int j = 0; j < NKEY; j++) {
			if (keyout.size() > 0) {
				bool ok = false;
				for (list<const char *>::const_iterator i = keyout.begin(); i != keyout.end(); i++) {
					if (!strcmp(*i, clean_keyword[j] ) ) ok = true;
				}
				if (ok)
					fprintf(fp, "%-13.5lf ", t[j]);
			}
			else {
				fprintf(fp, "%-13.5lf ", t[j]);
			}
		}
		if (cr) fprintf(fp, "\n");
	}
	int Print( FILE * fp, bool header = true ) {
		list<const char *> keyout;
		if (header)
			PrintHead(fp, keyout, true);
		PrintVal(fp, keyout, true);
		return 0;
	}
};


void wall_time_rec(const char * a, char * timestr) {
	while (strncmp(a, "CPU", 3)) a++;
	while (*a != ' ') a++;
	while (strncmp(a, "WALL", 4)) *timestr++ = *a++;
	*timestr = '\0';
}


double get_wtime(const char * line) {
	char timestr[16];
	wall_time_rec(line, timestr);
	if (debug) printf("timestr: %s\n", timestr);
	const int TSTRLEN = 16;
	char ore[TSTRLEN], minuti[TSTRLEN], secondi[TSTRLEN];
	char val[TSTRLEN];
	ore[0] = '\0';
	minuti[0] = '\0';
	secondi[0] = '\0';
	int c = 0, t = 0;
	while (timestr[t] == ' ') t++;
	while (timestr[t] != '\0' && c < (TSTRLEN-2)) {
		if (timestr[t] == 'h') {
			val[c] = '\0';
			strncpy(ore, val, TSTRLEN);
			t++; c = 0;
		}
		else if (timestr[t] == 'm') {
			val[c] = '\0';
			strncpy(minuti, val, TSTRLEN);
			t++; c = 0;
		}
		val[c++] = timestr[t++];
	}
	val[c] = '\0';
	strncpy(secondi, val, TSTRLEN);

	double o = atof(ore);
	double m = atof(minuti);
	double s = atof(secondi);
	if (debug) printf("#%s# %lf - #%s# %lf - #%s# %lf\n", ore, o, minuti, m, secondi, s);

	return 3600.0*o + 60.0*m + s;

}


int parsefile(FILE * fp, list<pair<ParaGeom ,Timing > > & table ) {
	char line[1024];
	ParaGeom run;
	Timing wtime;
	while (!feof(fp)) {
		fgets(line, 1023, fp);
		run.Parse(line);
		wtime.GetKey(line);
	}
	table.push_back(make_pair(run, wtime));
	return 0;
}


int main(int argc, char ** argv) {

	list <pair<ParaGeom, Timing> > table;
	list <const char * > pw_out;
	list <const char * > key_out;

	int j = 0;
	while ( ++j < argc ) {
		if (!strcmp(argv[j],"-key")) {
			key_out.push_back(argv[++j]);
		}
		else {
			pw_out.push_back(argv[j]);
		}
	}

	for (int j = 0; j < NKEY; j++) {
		strcpy(clean_keyword[j], CleanKey(keyword[j]) ); 
	}

	for (list <const char * >::const_iterator i = pw_out.begin(); i != pw_out.end(); i++) {
		FILE * in = fopen(*i, "r");
		if( !in ) {
			fprintf(stderr,"ERROR opening file %s\n", *i );
			exit(1);
		}
		printf("processing %s\n", *i);
		parsefile(in,table);
		fclose(in);
	}

	for (list <pair<ParaGeom, Timing> >::iterator t = table.begin(); t != table.end(); t++) {
		if (t == table.begin()) {
			t->first.PrintHead(stdout);
			t->second.PrintHead(stdout, key_out);
			printf("\n");
		}
		t->first.PrintVal(stdout);
		t->second.PrintVal(stdout, key_out);
		printf("\n");
	}

	return 0;

}

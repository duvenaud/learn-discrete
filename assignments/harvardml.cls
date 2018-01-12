% Ryan Adams
% School of Engineering and Applied Sciences
% Harvard University
% v0.01, 31 August 2013
% Based on HMC Math Dept. template by Eric J. Malm.
\NeedsTeXFormat{LaTeX2e}[1995/01/01]
\ProvidesClass{harvardml}
[2013/08/31 v0.01 Harvard ML Assignment Class]

\RequirePackage{ifpdf}

\newif\ifhmlset@submit
\DeclareOption{submit}{%
  \hmlset@submittrue%
}
\DeclareOption{nosubmit}{%
  \hmlset@submitfalse%
}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ExecuteOptions{nosubmit}
\ProcessOptions\relax

\LoadClass[10pt,letterpaper]{article}

\newif\ifhmlset@header

\hmlset@headertrue

\RequirePackage{mathpazo}
\RequirePackage{palatino}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{amsthm}
\RequirePackage{fullpage}
\RequirePackage{mdframed}

\newtheoremstyle{hmlplain}
                {3pt}% Space above
                {3pt}% Space below
                {}% Body font
                {}% Indent amount
                {\bfseries}% Theorem head font
                {\\*[3pt]}% Punctuation after theorem head
                {.5em}% Space after theorem head
                {}% Theorem head spec (can be left empty, meaning `normal')

\def\titlebar{\hrule height2pt\vskip .25in\vskip-\parskip}

\newcommand{\headerblock}{%
  \noindent\begin{minipage}{0.33\textwidth}
  \begin{flushleft}
  \ifhmlset@submit
    \mbox{\hmlset@name}\\
    \mbox{\tt \hmlset@email}\\
    \mbox{\hmlset@course}
  \fi
  \end{flushleft}
  \end{minipage}
  \noindent\begin{minipage}{0.33\textwidth}
  \begin{center}
    \mbox{\Large\hmlset@assignment}\protect\\
    %Due: \hmlset@duedate
  \end{center}
  \end{minipage}
  \noindent\begin{minipage}{0.33\textwidth}
  \begin{flushright}
  \ifhmlset@submit
  %  %Collaborators: \hmlset@collaborators
  \fi
  \end{flushright}
  \end{minipage}
  \vspace{0.1cm}
  \titlebar
}

\ifhmlset@header\AtBeginDocument{\headerblock}\fi

\def\hmlset@name{}
\def\hmlset@email{}
\def\hmlset@course{}
\def\hmlset@assignment{}
\def\hmlset@duedate{}
\def\hmlset@collaborators{}
\def\hmlset@extraline{}

% commands to set header block info
\newcommand{\name}[1]{\def\hmlset@name{#1}}
\newcommand{\email}[1]{\def\hmlset@email{#1}}
\newcommand{\course}[1]{\def\hmlset@course{#1}}
\newcommand{\assignment}[1]{\def\hmlset@assignment{#1}}
\newcommand{\duedate}[1]{\def\hmlset@duedate{#1}}
\newcommand{\collaborators}[1]{\def\hmlset@collaborators{#1}}
\newcommand{\extraline}[1]{\def\hmlset@extraline{#1}}

\theoremstyle{hmlplain}
\newmdtheoremenv[skipabove=\topsep,skipbelow=\topsep,nobreak=true]{problem}{Problem}

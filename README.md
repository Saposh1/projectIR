# $\color{lightblue}{Wikipedia\ IR\ Engine}$

<p align="center">
  <img width="250" height="250" src="https://upload.wikimedia.org/wikipedia/commons/9/98/New-Bouncywikilogo.gif?20220824134145">
</p>

$\color{#db5a6b}{In\ the\ following\ project,\ we\ created\ a\ retrieval\ engine\ that\ is\ working\ on\ the\ english\ wikipedia\ updated\ to\ august\ 2021.}$
$\color{#db5a6b}{The\ main\ propose\ of\ the\ project\ -\ maximize\ the\ map@40\ of\ our\ predicted\ results.}$
$\color{#db5a6b}{In\ order\ to\ do\ so,\ we\ used\ weighted\ bm25\ algorithm\ and\ weighted\ titles\ of\ the\ wikipedia\ pages.}$
$\color{#db5a6b}{In\ addition,\ we\ added\ page\ rank\ score\ to\ each\ candidate\ document\ in\ order\ to\ boost\ relevant\ docs\ only.}$

$\color{#c17b69}{Attached\ files\ description:}$<br />
$\color{#560319}{createdInvertedIndexGCP.ipynb\ -\ This\ jupyter\ notebook\ was\ used\ to\ create\ the\ inverted\ indexes.\ }$<br />
$\color{#560319}{inverted\\_index\\_gcp.py\ -\ This\ python\ file\ was\ used\ to\ create\ the\ inverted\ indexes.\ }$<br />
$\color{#560319}{paths\ to\ anchor\ bins.txt\ -\ This\ txt\ file\ includes\ all\ the\ paths\ to\ bin\ and\ pkl\ files\ of\ anchor\ inverted\ index.}$<br />
$\color{#560319}{paths\ to\ body\ bin.txt\ -\ This\ txt\ file\ includes\ all\ the\ paths\ to\ bin\ and\ pkl\ files\ of\ body\ inverted\ index.}$<br />
$\color{#560319}{paths\ to\ titles\ bins.txt\ -\ This\ txt\ file\ includes\ all\ the\ paths\ to\ bin\ and\ pkl\ files\ of\ title\ inverted\ index.}$<br />
$\color{#560319}{run\\_frontend\\_in\\_gcp.sh\ -\ This\ script\ file\ was\ used\ to\ create\ an\ instance\ in\ google\ cloud.\ }$<br />
$\color{#560319}{search\\_backend.py\ -\ This\ python\ file\ includes\ all\ the\ implementations\ of\ our\ engine\ retrieve\ methods.}$<br />
$\color{#560319}{search\\_frontend.py\ -\ Includes\ 6\ methods\ of\ retrieval:\ search,\ search\\_body,\ search\\_title,\ search\\_anchor,\ get\\_pagerank,\ get\\_pageview.}$
$\color{#560319}{startup\\_script\\_gcp.sh\ -\ This\ script\ file\ was\ used\ to\ install\ all\ the\ necessary\ python\ libs.\ }$<br />

$\color{#c17b69}{Hopefully\ you\ will\ enjoy\ this\ engine\ as\ much\ as\ we\ did!\}$




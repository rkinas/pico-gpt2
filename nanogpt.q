/ nanogpt.q — GPT-2 in q/kdb+ (port of nanogpt.py)
\S 42
system"test -f input.txt||curl -so input.txt https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"
D:read0`:input.txt;D:D@neg[c]?c:#D;u:asc distinct raze D;nb:#u;nz:nb+1
vd::vg::0#0.;vc::vl::()
nv:{i:#vd;vd::vd,x;vg::vg,0.;vc::vc,enlist y;vl::vl,enlist z;i};lf:{nv[x;`long$();`float$()]}
ad:{nv[vd[x]+vd y;x,y;1 1.]};ml:{nv[vd[x]*vd y;x,y;(vd y;vd x)]}
pw:{nv[xexp[vd x;y];enlist x;enlist y*xexp[vd x;y-1]]};ex:{e:exp vd x;nv[e;enlist x;enlist e]}
lg:{nv[log vd x;enlist x;enlist 1%vd x]};rl:{nv[0.|vd x;enlist x;enlist 0.+vd[x]>0]};dv:{ml[x;pw[y;-1.]]}
bw:{bt::bs::`long$();df:{$[x in bs;;[bs::bs,x;df'vc x;bt::bt,x]]};df x;vg[x]::1.;{vg[vc x]::vg[vc x]+vl[x]*vg x}'|bt}
E:16;H:4;W:16;R:E div H;mx:{y cut lf'.08*-1+2*(x*y)?1.}
S:`e`p`h`q`k`v`o`f`b!(mx[nz;E];mx[W;E];mx[nz;E];mx[E;E];mx[E;E];mx[E;E];mx[E;E];mx[4*E;E];mx[E;4*E])
P:raze raze'value S;NP:#vd;lr:.01;b1:.85;b2:.99;ea:1e-8;AM:AN:(#P)#0.
F:{[x;w]{ad/ml'[y;x]}'w};X:{e:ex'ad'[x;(#x)#lf neg max vd x];t:ad/e;dv'[e;(#e)#t]};J:{c:pw[ad[dv[ad/ml'[x;x];lf 0.+#x];lf 1e-5];-.5];ml'[x;(#x)#c]}
KK::CC::()
G:{[i;p]x:J ad'[S[`e]i;S[`p]p];y:x;x:J x;qr:F[x;S`q];kr:F[x;S`k];vr:F[x;S`v];KK::KK,enlist kr;CC::CC,enlist vr;a:raze{[qr;h]ix:h*R+til R;qh:qr@ix;sc:X{[qh;kt]dv[ad/ml'[qh;kt];lf R xexp .5]}[qh]'KK[;ix];{[sc;vt]ad/ml'[sc;vt]}[sc]'flip CC[;ix]}[qr]'til H;x:ad'[F[a;S`o];y];y:x;x:rl'F[J x;S`f];x:ad'[F[x;S`b];y];F[x;S`h]}
{[s]d:D s mod #D;t:nb,(u?d),nb;n:W&-1+#t;vd::NP#vd;vg::NP#0.;vc::NP#vc;vl::NP#vl;KK::CC::();L:ml[dv[ad/{lg X[G[t x;x]]t x+1}'til n;lf 0.+n];lf -1.];bw L;z:lr*1-s%1000;g:vg P;AM::b1*AM+(1-b1)*g;AN::b2*AN+(1-b2)*g*g;vd[P]::vd[P]-z*(AM%(1-xexp[b1;s+1]))%ea+sqrt AN%(1-xexp[b2;s+1]);-1"step ",(-4)$string 1+s," / 1000 | loss ",.Q.f[4;vd L]}'til 1000
-1"\n--- inference (new, hallucinated names) ---"
ws:{sum(sums x%sum x)<first 1?1.};gen:{[t;p;s]lo:G[t;p];pr:vd X ml'[lo;(#lo)#lf 2.];n:ws pr;$[(n=nb)|p>=W-1;s;gen[n;p+1;s,u n]]}
{vd::NP#vd;vg::NP#0.;vc::NP#vc;vl::NP#vl;KK::CC::();-1"sample ",(-2)$string 1+x,": ",gen[nb;0;""]}'til 20

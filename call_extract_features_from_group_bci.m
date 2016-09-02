function call_extract_features_from_group_bci(fn,dt) % ver 1 14Aug 
  % BCI EEG Data http://bnci-horizon-2020.eu/database/data-sets
  % fn is the apath to bci_eeg mat files, eg fn='/home/vs/Dropbox/Matlab/eeg_bci/mat/'
  % dt={1,2}: 1 for all train "T" and 2 for all test "E" mat files
  % 
  global D2 m1 noft 
  call_settings_bci();
  if dt==1
    dt1='T';  
  elseif dt==2
    dt1='E';
  end
  fn1=sprintf('%sS%s%s.mat',fn,'*',dt1); 
  Df=dir(fn1);
  nof=size(Df,1); % nof subjects
  noft1=noft*5;
  A=1:noft1;
  XY=zeros(nof*noft1,m1+1);
  for i=1:nof
    fnam=sprintf('%sS%02i%s.mat',fn,i,dt1); 
    if exist(fnam,'file')
      fn1=[fn Df(i).name];
      fprintf('.eeg_feature_from %s\n',fn1)
      D1=load(fn1);
      D2=D1(1).data;
      XY1=call_eeg_trials();
      XY(A,:)=XY1;
      A=A+noft1;
    end
  end
  wf=['extract_feature_bci_' dt1];
  save(wf,'XY');
  wf=[wf '.csv'];
  csvwrite(wf,XY);
return


function call_settings_bci() % 24Aug
  global Fs B1 m1 nob noc noft 
  %addpath('my_lib')
  % https://en.wikipedia.org/wiki/Electroencephalography
  % Delta: <4
  % Theta: 4-7
  % Alpha: 8-15
  % Beta: 16-31
  % Gamma: >32
  % Mu: 8-12
  B1=[0 4; 4 7; 7 15; 15 31; 31 50; 8 12]; % the above 6 frequency bands 
  nob=size(B1,1);
  noc=15;
  m1=noc*(3*nob+nob*(nob-1)/2+2);
  noft=20; % nof trials
  Fs=512;     % sampling rate
  % Wikipedia\Gamma_wave 
  %Fgammamax=50;% max frequency 
return


function Tr=call_eeg_trials() % classes {right_hand,feet}
  global D2 Dx m1 len1 noft
  Tr=zeros(5*noft,m1+1);
  k=0;
  for i=1:5         % trials 1-5
    D=D2{1,i};
    i1=0;
    for j=1:noft    % trials 1-20
      A=(i1+1):D.trial(j);
      len1=my_lib_power2(length(A)); % my_lib
      A=A(1:len1);
      Dx=D.X(A,:);
      X=call_spectral_bands_features();
      k=k+1;
      Tr(k,:)=[X D.y(j)];
      i1=D.trial(j);
    end
  end
return


function n2=my_lib_power2(len)
  n=log2(len);
  n1=fix(n);
  n2=2^n1;
return


function P=call_spectral_bands_features()
  global nob noc 
  [P,F]=call_psd();                   % CALL
  [P1,P2,P3,Ps,Es]=calc_pow_band(P,F);   % CALL
  P4=zeros(nob*(nob-1)/2,noc); % ratios for abs_power P1
  for j=1:noc
    k=0;
    for i1=1:nob-1
      for i2=i1+1:nob
        k=k+1;
        P4(k,j)=P1(i1,j)/P1(i2,j);
      end
    end
  end
  % vectorization  
  P=[P1(:)' P2(:)' P3(:)' P4(:)' Ps Es];
return


function [P,F]=call_psd()
  % http://uk.mathworks.com/help/signal/ug/psd-estimate-using-fft.html
  % P is N/2+1:tilenum power spectral density
  %
  global Fs Dx len1
  N=len1; 
  Xdft=fft(Dx); % Dx is a matrix with Spectral powers per columns
  Xdft=Xdft(1:N/2+1,:);         % for even length signals add 1
  P=(1/(Fs*N))*abs(Xdft).^2; % ^2 for power
  P(2:end-1,:)=2*P(2:end-1,:);
  F=0:Fs/N:Fs/2;
return


function [P1,P2,P3,Ps,E1]=calc_pow_band(P,F) 
  %   P1 m1-abs spectral powers 
  %   P2 m1-rel P1
  %   P3 m1-var P1
  %   Ps total spectral power
  %   E spectral entropy
  %
  global lenb1 B1 lenp 
  lenp=size(P,2);
  P1=zeros(lenb1,lenp);     
  P3=P1;
  for i=1:lenb1
    V=F>B1(i,1) & F<=B1(i,2);
    P1(i,:)=mean(P(V,:));    % was sum
    P3(i,:)=var(P(V,:));     % varience
  end
  Ps=sum(P1);
  En=zeros(lenb1,lenp);     % spectral En 
  P2=zeros(lenb1,lenp);
  for i=1:lenb1
    P2(i,:)=P1(i,:)./Ps;
    En(i,:)=P2(i,:).*log(P2(i,:));
  end
  E1=-sum(En);
return


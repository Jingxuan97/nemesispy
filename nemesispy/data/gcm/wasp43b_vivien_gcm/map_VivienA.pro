print,'Plot to X-window(0) or eps(1) : '
ips=0
read,ips

if(ips eq 0) then begin
 set_plot,'X'
 device,retain=2,decomposed=0
 lx=1.
 window,0,xsize=700,ysize=700
endif else begin
 set_plot,'ps'
 device,filename='map_vivienA.eps',/encapsulated,/color,xsize=18,ysize=10
endelse

nlon=1
nlat=1

openr,1,'process_vivien.txt'
readf,1,nlon,nlat
xlon=fltarr(nlon)
xlat=fltarr(nlat)
readf,1,xlon
readf,1,xlat
ydisc = sin(xlat*!pi/180)
npv=1
readf,1,npv
pv=fltarr(npv)
readf,1,pv

tmp=fltarr(7,npv)
tmap=fltarr(nlon,nlat,npv)
co2map=fltarr(nlon,nlat,npv)
h2map=fltarr(nlon,nlat,npv)
hemap=fltarr(nlon,nlat,npv)
ch4map=fltarr(nlon,nlat,npv)
comap=fltarr(nlon,nlat,npv)
h2omap=fltarr(nlon,nlat,npv)

for ilon=0,nlon-1 do for ilat=0,nlat-1 do begin
 readf,1,tmp
 tmap(ilon,ilat,*)=tmp(0,*)
 co2map(ilon,ilat,*)=tmp(1,*)
 h2map(ilon,ilat,*)=tmp(2,*)
 hemap(ilon,ilat,*)=tmp(3,*)
 ch4map(ilon,ilat,*)=tmp(4,*)
 comap(ilon,ilat,*)=tmp(5,*)
 h2omap(ilon,ilat,*)=tmp(6,*)
endfor

close,1


ioff=12+2*findgen(6)
clev = 500+100*findgen(18)

tmapx = fltarr(nlon,npv)
tmapy = fltarr(nlon,npv)
sumw=0.
sumw = total(cos(xlat*!pi/180.))
for i=0,nlon-1 do begin
 tmapx(i,*)=0.5*(tmap(i,15,*)+tmap(i,16,*))
 for j=0,npv-1 do begin
  sum=0.
  for k=0,nlat-1 do sum=sum+tmap(i,k,j)*cos(xlat(k)*!pi/180.)
  tmapy(i,j)=sum/sumw
 endfor
endfor


!p.multi=[0,2,3]


yt = [-60.,-30.,0,30.,60.]
yn = ['-60','-30','0','30','60']
yv = sin(yt*!pi/180.)

for ilev = 0, 5 do begin

 jk = ioff(ilev)

 loadct,0

 t1 = fltarr(nlon,nlat)
 for i=0,nlon-1 do for j=0,nlat-1 do t1(i,j)=tmap(i,j,jk)

 contour,t1,xlon,ydisc,yrange=[-1,1],ystyle=1,xrange=[-180,180],$
  xstyle=1,xtitle='Longitude',ytitle='N/S',levels=clev,xticks=12,ytickv=yv,ytickname=yn,$
  title=strcompress(string('Pressure : ',pv(jk))),yticks=4,thick=lx,ythick=lx,xthick=lx

 loadct,72
 stretch,255,0

 contour,t1,xlon,ydisc,yrange=[-1,1],ystyle=1,xrange=[-180,180],$
  xstyle=1,xtitle='Longitude',ytitle='N/S',levels=clev,/fill,$
  /overplot,xticks=12,ytickv=yv,ytickname=yn,yticks=4,thick=lx,ythick=lx,xthick=lx

 loadct,0
  contour,t1,xlon,ydisc,yrange=[-1,1],ystyle=1,xrange=[-180,180],$
  xstyle=1,xtitle='Longitude',ytitle='N/S',levels=clev,xticks=12,$
  /follow,/overplot,ytickv=yv,ytickname=yn,yticks=4,thick=lx,ythick=lx,xthick=lx

endfor

if(ips gt 0) then begin
 device,/close
 set_plot,'X'
endif

end

/*	Win32 Portions of the chart window class		Sergei L. Kosakovsky Pond, Spring 2000 - December 2002.*/#include "HYChartWindow.h"#include "HYCanvas.h"#include "HYUtils.h"#include "HYPulldown.h"#include "HYDialogs.h"#include "math.h"extern	 _Parameter 			   pi_const;#define	 HY_CHART_WIN32_MENU_BASE   6000#define	 HY_CHARTD_WIN32_MENU_BASE  27000//__________________________________________________________________void _HYChartWindow::_SetMenuBar(void){	_HYWindow::_SetMenuBar();	HMENU     		 windowMenu = GetMenu (theWindow),					 editMenu   = GetSubMenu(windowMenu,1),		   	   		 chartMenu  = GetSubMenu(windowMenu,2);		if (!chartMenu)	{		chartMenu = CreateMenu();					HMENU	       saveMenu  =  CreatePopupMenu(),				       printMenu =  CreatePopupMenu(),				       fontMenu  =  CreatePopupMenu(),				       procMenu  =  CreatePopupMenu();				       		EnableMenuItem (editMenu,2,MF_BYPOSITION|MF_ENABLED);		EnableMenuItem (editMenu,4,MF_BYPOSITION|MF_ENABLED);		EnableMenuItem (editMenu,6,MF_BYPOSITION|MF_ENABLED);				   		checkPointer  (saveMenu);		checkPointer  (chartMenu);		checkPointer  (fontMenu);		checkPointer  (printMenu);		checkPointer  (procMenu);				InsertMenu 	 	(chartMenu, 0xFFFFFFFF, MF_BYPOSITION|MF_STRING, HY_CHART_WIN32_MENU_BASE+4, "Chart &Name");				  		InsertMenu 	 	(chartMenu, 0xFFFFFFFF, MF_BYPOSITION|MF_STRING, HY_CHART_WIN32_MENU_BASE, "Chart &Options");		InsertMenu 	 	(saveMenu,  0xFFFFFFFF, MF_BYPOSITION|MF_STRING, HY_WINDOW_MENU_ID_FILE+1, "Save &Chart\tCtrl-S");		InsertMenu 	 	(saveMenu,  0xFFFFFFFF, MF_BYPOSITION|MF_STRING, HY_WINDOW_MENU_ID_FILE+3, "Save &Graphic");		InsertMenu 	 	(saveMenu,  0xFFFFFFFF, MF_BYPOSITION|MF_STRING, HY_WINDOW_MENU_ID_FILE+4, "Save &Table");		InsertMenu 	 	(printMenu, 0xFFFFFFFF, MF_BYPOSITION|MF_STRING, HY_WINDOW_MENU_ID_FILE+2, "Print &Graphic\tCtrl-P");		InsertMenu 	 	(printMenu, 0xFFFFFFFF, MF_BYPOSITION|MF_STRING, HY_WINDOW_MENU_ID_FILE+5, "Print &Data");		InsertMenu 	 	(fontMenu,  0xFFFFFFFF, MF_BYPOSITION|MF_STRING, HY_CHART_WIN32_MENU_BASE+1, "&Tickmark  Font");		InsertMenu 	 	(fontMenu,  0xFFFFFFFF, MF_BYPOSITION|MF_STRING, HY_CHART_WIN32_MENU_BASE+2, "&Legend Font");		InsertMenu 	 	(fontMenu,  0xFFFFFFFF, MF_BYPOSITION|MF_STRING, HY_CHART_WIN32_MENU_BASE+3, "&Axis Label  Font");				InsertMenu 	 	(chartMenu, 0xFFFFFFFF, MF_BYPOSITION|MF_POPUP, (UINT)fontMenu, "&Fonts");		InsertMenu 	 	(chartMenu, 0xFFFFFFFF, MF_BYPOSITION|MF_SEPARATOR, 0, nil);		if (chartProcessors.lLength == 0)		{			InsertMenu 	 	(chartMenu, 0xFFFFFFFF, MF_BYPOSITION|MF_STRING|MF_GRAYED, 0, "&Data Processing");			DestroyMenu 	(procMenu);		}		else		{			for (long k=0; k<chartProcessors.lLength; k++)			{				_String *thisItem = (_String*)chartProcessors (k), 						chopped = thisItem->Cut (thisItem->FindBackwards ('\\',0,-1)+1,-1);								InsertMenu 	 	(procMenu, 0xFFFFFFFF, MF_BYPOSITION|MF_STRING, HY_CHART_WIN32_MENU_BASE+5+k, chopped.sData);			}			InsertMenu 	 	(chartMenu, 0xFFFFFFFF, MF_BYPOSITION|MF_POPUP, (UINT)procMenu, "&Data Processing");		}				InsertMenu	 (windowMenu, 2, MF_BYPOSITION|MF_POPUP, (UINT) chartMenu , "&Chart");				chartMenu =  GetSubMenu(windowMenu,0);						ModifyMenu	 (chartMenu, 0, MF_BYPOSITION|MF_POPUP, (UINT) saveMenu , "&Save");		ModifyMenu	 (chartMenu, 1, MF_BYPOSITION|MF_POPUP, (UINT) printMenu , "&Print");				_AddStandardAccels();		_BuildAccelTable  (true);			accels.Clear();	}}//__________________________________________________________________void _HYChartWindow::_UnsetMenuBar(void){}//__________________________________________________________________void 		_HYChartWindow::_PrintChart(void){			DOCINFO  				di = {sizeof(DOCINFO), "HYPHY.out", NULL };	PRINTDLG 				pd;	BOOL            		SuccessFlag;		pd.lStructSize         = sizeof(PRINTDLG);	pd.hwndOwner           = theWindow;	pd.hDevMode            = NULL;	pd.hDevNames           = NULL;	pd.hDC                 = NULL;					pd.Flags               = PD_COLLATE | PD_RETURNDC | PD_NOSELECTION;	pd.nFromPage           = 1;						pd.nToPage             = 0xffff;				pd.nMinPage            = 1;	pd.nMaxPage            = 0xffff;				pd.nCopies             = 1;	pd.hInstance           = NULL;	pd.lCustData           = 0L;	pd.lpfnPrintHook       = NULL;	pd.lpfnSetupHook       = NULL;	pd.lpPrintTemplateName = NULL;	pd.lpSetupTemplateName = NULL;	pd.hPrintTemplate      = NULL;	pd.hSetupTemplate      = NULL;		if (!PrintDlg(&pd))											return;										if (pd.hDC == NULL) 										pd.hDC = GetPrinterDeviceContext(theWindow);		EnableWindow(theWindow, FALSE);	SuccessFlag   = TRUE;	UserAbortFlag = FALSE;	PrintDialogHandle = CreateDialog(GetModuleHandle(NULL), (LPCTSTR)"PrintDlgBox", theWindow, 																PrintDialogProc);	SetDlgItemText(PrintDialogHandle, IDD_FNAME, "Chart Printing...");	SetAbortProc(pd.hDC, AbortProc);	if (StartDoc(pd.hDC, &di) > 0)	{		HDC			windowDC = GetDC (theWindow);				long		printW = GetDeviceCaps(pd.hDC, HORZRES), 					printH = GetDeviceCaps(pd.hDC, VERTRES),										hRes = GetDeviceCaps(pd.hDC, LOGPIXELSX),					vRes = GetDeviceCaps(pd.hDC, LOGPIXELSY),										screenHRes = GetDeviceCaps(windowDC, LOGPIXELSX),					screenVRes = GetDeviceCaps(windowDC, LOGPIXELSY),									 	fromPage = pd.nMinPage,				 	toPage   = pd.nMaxPage;				 					 		if (pd.Flags & PD_PAGENUMS)		{			fromPage = pd.nFromPage;			toPage   = pd.nToPage;		}				ReleaseDC   (theWindow, windowDC);		hRes = printW*((_Parameter)screenHRes/hRes);		vRes = printH*((_Parameter)screenVRes/vRes);		screenHRes = printW;		screenVRes = printH;		printW = hRes;		printH = vRes;				if (StartPage (pd.hDC) <= 0)		{			SuccessFlag = FALSE;		}		else		{			SetMapMode	(pd.hDC, MM_ISOTROPIC);		    SetWindowExtEx (pd.hDC, hRes, vRes,nil);			SetViewportExtEx (pd.hDC, screenHRes, screenVRes, nil);						_HYRect		viewRect  = ((_HYStretchCanvas*)GetObject (0))->GetCanvasSize();						_Parameter	aspectRatio = viewRect.right/(_Parameter)viewRect.bottom;						_HYRect 	printRect = {0,0,printH, printH*aspectRatio,0};						if (printRect.right > printW)			{				aspectRatio = printW/(_Parameter)printRect.right;				printRect.right = printW;				printRect.bottom *= aspectRatio;			}								 			_HYStretchCanvas    *sc = (_HYStretchCanvas*)GetObject (0);			HDC			saveDC = sc->thePane;						sc->thePane = pd.hDC;			DrawChart	(&printRect);			sc->thePane = saveDC;						if (EndPage (pd.hDC) <= 0)				SuccessFlag = FALSE;		}	}	else		SuccessFlag = FALSE;	if (SuccessFlag)		SuccessFlag = (EndDoc(pd.hDC)>0);	if (!UserAbortFlag)	{		EnableWindow(theWindow, TRUE);		DestroyWindow(PrintDialogHandle);	}	DeleteDC (pd.hDC);	if (!SuccessFlag && !UserAbortFlag)	{		_String errMsg = _String("Failed to print the chart. Windows Error:") & (long)GetLastError();		ProblemReport (errMsg,nil);	}}//__________________________________________________________________bool 		_HYChartWindow::_ProcessMenuSelection (long msel){		switch (msel)	{		case HY_CHART_WIN32_MENU_BASE: // chart menu		{			HandleChartOptions ();			return true;		}			case HY_WINDOW_MENU_ID_FILE+1: // save menu		case HY_WINDOW_MENU_ID_FILE+3: // save menu		case HY_WINDOW_MENU_ID_FILE+4: // save menu		{			DoSave ((msel==HY_WINDOW_MENU_ID_FILE-1)?0:msel-HY_WINDOW_MENU_ID_FILE-2);			return true;		}			case HY_WINDOW_MENU_ID_FILE+2: // print menu		case HY_WINDOW_MENU_ID_FILE+5: // print menu		{			DoPrint ((msel==HY_WINDOW_MENU_ID_FILE+2)?0:-1);			return true;		}			case HY_CHART_WIN32_MENU_BASE+1: // font menu		case HY_CHART_WIN32_MENU_BASE+2: // font menu		case HY_CHART_WIN32_MENU_BASE+3: // font menu		{			DoChangeFont (msel-HY_CHART_WIN32_MENU_BASE-1);			return true;		}			case HY_CHART_WIN32_MENU_BASE+4: // chart name		{			RenameChartWindow ();			return true;		}		default: // proc menu		{			if (msel>=HY_CHART_WIN32_MENU_BASE+5)			{				ExecuteProcessor (msel-HY_CHART_WIN32_MENU_BASE-5);				return true;			}		}		}	return _HYTWindow::_ProcessMenuSelection(msel);}//__________________________________________________________________bool _HYChartWindow::_ProcessOSEvent (Ptr vEvent){	static int   lastH = -1,				 lastV = -1;				 	if (!_HYTWindow::_ProcessOSEvent (vEvent))	{			_HYWindowsUIMessage*	theEvent = (_HYWindowsUIMessage*)vEvent;				if (components.lLength == 0) 			return false;				_HYPullDown *p1 = (_HYPullDown*)GetObject (4);						if (p1&&(p1->GetSelection()>=8)&&(ySeries.lLength))		{			switch (theEvent->iMsg)			{				case WM_LBUTTONDOWN:				{					lastH = (short)LOWORD (theEvent->lParam);					lastV = (short)HIWORD (theEvent->lParam);															if (FindClickedCell(lastH, lastV)!=0) // the chart					{						lastH = -1;						lastV = -1;					}					else					{						SetCapture (theWindow);						return 		true;					}					break;				}								case WM_LBUTTONUP:				{					if (lastH>=0)					{						ReleaseCapture ();						lastH = -1;						lastV = -1;						return  true;					}					break;				}								case WM_MOUSEMOVE:				{					if (lastH>=0)					{						short 		newH = (short)LOWORD (theEvent->lParam),							  		newV = (short)HIWORD (theEvent->lParam);												bool 	 	redraw = false;												_Parameter  stepper = pi_const/180.;												if (abs(newH-lastH)>abs(newV-lastV))						{							stepper *= 1+log (fabs(newH-lastH))/log(2.0);							if (newH-lastH<0)							{								if (xyAngle>0.0)								{									xyAngle -= stepper;									if (xyAngle<0) 										xyAngle = 0;									redraw = true;								}							}							else								if (xyAngle<pi_const/2)								{									xyAngle += stepper;									if (xyAngle>pi_const/2) 										xyAngle = pi_const/2;									redraw = true;								}						}						else						{							if (newV==lastV)								return false;							stepper *= 1+log (fabs(newV-lastV))/log(2.0);							if (newV-lastV>0)							{								if (zAngle<pi_const/2)								{									zAngle += stepper;									if (zAngle>pi_const/2) 										zAngle = pi_const/2;									redraw = true;								}							}							else								if (zAngle>0.0)								{									zAngle -= stepper;									if (zAngle<0) 										zAngle = 0;									redraw = true;								}												}						if (redraw)						{							ComputeProjectionSettings();							projectionMatrix = ComputeProjectionMatrix	 ();							forceUpdateForScrolling = true;							DrawChart();							forceUpdateForScrolling = false;						}													lastH = newH;						lastV = newV;									}					break;				}			}		}		return false;	}	return true;}//__________________________________________________________________void _HYChartWindow::_CopyChart (void){	_HYStretchCanvas    *sc = (_HYStretchCanvas*)GetObject (0);	PlaceBitmapInClipboard (sc->paneBitMap, theWindow);}//__________________________________________________________________void _HYDistributionChartWindow::_SetMenuBar(void){	HMENU     		 chartMenu  = GetSubMenu(windowMenu,3);		if (!chartMenu)	{				chartMenu = CreateMenu();							InsertMenu 	 	(chartMenu, 0xFFFFFFFF, MF_BYPOSITION|MF_STRING , HY_CHARTD_WIN32_MENU_BASE, "Define New &Variable");				  		InsertMenu 	 	(chartMenu, 0xFFFFFFFF, MF_BYPOSITION|MF_STRING , HY_CHARTD_WIN32_MENU_BASE+1, "&Delete Variable");		InsertMenu 	 	(chartMenu,  0xFFFFFFFF, MF_BYPOSITION|MF_STRING, HY_CHARTD_WIN32_MENU_BASE+2, "&Conditional Distribution");		if (distribProcessors.lLength > 0)		{			InsertMenu 	 	(chartMenu, 0xFFFFFFFF, MF_BYPOSITION|MF_SEPARATOR, 0, nil);			for (long k=0; k<distribProcessors.lLength; k++)			{				_String *thisItem = (_String*)distribProcessors (k), 						chopped = thisItem->Cut (thisItem->FindBackwards ('\\',0,-1)+1,-1);								InsertMenu 	 	(chartMenu, 0xFFFFFFFF, MF_BYPOSITION|MF_STRING, HY_CHARTD_WIN32_MENU_BASE+3+k, chopped.sData);			}		}				InsertMenu	 (windowMenu, 3, MF_BYPOSITION|MF_POPUP, (UINT) chartMenu , "Cate&gories");	}}//__________________________________________________________________void _HYDistributionChartWindow::_UnsetMenuBar(void){	_HYChartWindow::_UnsetMenuBar();}//__________________________________________________________________bool _HYDistributionChartWindow::_ProcessMenuSelection (long msel){	switch (msel)	{		case HY_CHARTD_WIN32_MENU_BASE: 		{			AddVariable ();			return true;		}			case HY_CHARTD_WIN32_MENU_BASE+1: 		{			RemoveVariable ();			return true;		}			case HY_CHARTD_WIN32_MENU_BASE+2: 		{			ShowMarginals ();			return true;		}			default: 		{			if (msel>=HY_CHARTD_WIN32_MENU_BASE+3)			{				HandleCatPostProcessor (msel-HY_CHARTD_WIN32_MENU_BASE-3);				return true;			}		}		}	return _HYChartWindow::_ProcessMenuSelection(msel);}//EOF
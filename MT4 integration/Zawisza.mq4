#property copyright "Jaroslaw Mrugala"
#property link      "https://www.mrugalastudio.com/"
#property version   "1.00"
#property strict
#include <Zmq/Zmq.mqh>

extern string PROJECT_NAME = "Zawisza";
extern string ZEROMQ_PROTOCOL = "tcp";
extern string HOSTNAME = "127.0.0.1";
extern int REP_PORT = 5555;
extern int PUSH_PORT = 5556;
extern int MILLISECOND_TIMER = 1;  // 1 millisecond

extern string t0 = "--- Trading Parameters ---";
extern int MagicNumber = 123465;
extern int MaximumOrders = 10;
extern double MaximumLotSize = 0.01;

// CREATE ZeroMQ Context
Context context(PROJECT_NAME);
// CREATE ZMQ_REP SOCKET
Socket repSocket(context,ZMQ_REP);
// CREATE ZMQ_PUSH SOCKET
Socket pushSocket(context,ZMQ_PUSH);
// VARIABLES FOR LATER
uchar data[];
ZmqMsg request;
string ret = "";


void OnTimer()
{
//recive, understand answer
   repSocket.recv(request,true);  
   ZmqMsg reply = MessageHandler(request);
   repSocket.send(reply);
   
  
///send tick data
/*
   ret = "N/A";  
   ret = GetBidAsk("EURUSD"); 
   InformPullClient(pushSocket, ret); 
   */
}

ZmqMsg MessageHandler(ZmqMsg &request) {
   
   // Output object
   ZmqMsg reply;
   
   // Message components for later.
   string components[];
   
   if(request.size() > 0) {
      ArrayResize(data, request.size()); // Get data from request
      request.getData(data);
      string dataStr = CharArrayToString(data);
      ParseZmqMessage(dataStr, components); // Process data
      
      
      Print(StringFormat("[ZAWISZA] Processing: %s", dataStr));
      ZmqMsg ret(StringFormat("[ZAWISZA] Processing: %s", dataStr)); // Construct response
      reply = ret;
      
      InterpretRychezaMessage(&pushSocket, components); // execute orders
      
   }
   else { // NO DATA RECEIVED
      ZmqMsg ret("[SERVER]- empty order"); // Construct response
      reply = ret;
   }
   
   return(reply);
}

// Interpret Zmq Message and perform actions
void InterpretRychezaMessage(Socket &pSocket, string& compArray[]) {

   Print("[Zawisza]: Interpreting Message..");
   
   // Message Structures:
   
   // 1) Trading
   // TRADE|ACTION|TYPE|SYMBOL|PRICE|SL|TP|COMMENT|TICKET
   // e.g. TRADE|OPEN|1|EURUSD|0|50|50|R-to-MetaTrader4|12345678
   
   // The 12345678 at the end is the ticket ID, for MODIFY and CLOSE.
   
   // 2) Data Requests
   
   // 2.1) RATES|SYMBOL   -> Returns Current Bid/Ask
   
   // 2.2) DATA|SYMBOL|TIMEFRAME|START_DATETIME|END_DATETIME
   
   // NOTE: datetime has format: D'2015.01.01 00:00'
   
   /*
      compArray[0] = TRADE or RATES
      If RATES -> compArray[1] = Symbol
      
      If TRADE ->
         compArray[0] = TRADE
         compArray[1] = ACTION (e.g. OPEN, MODIFY, CLOSE)
         compArray[2] = TYPE (e.g. OP_BUY, OP_SELL, etc - only used when ACTION=OPEN)
         
         // ORDER TYPES: 
         // https://docs.mql4.com/constants/tradingconstants/orderproperties
         
         // OP_BUY = 0
         // OP_SELL = 1
         // OP_BUYLIMIT = 2
         // OP_SELLLIMIT = 3
         // OP_BUYSTOP = 4
         // OP_SELLSTOP = 5
         
         compArray[3] = Symbol (e.g. EURUSD, etc.)
         compArray[4] = Open/Close Price (ignored if ACTION = MODIFY)
         compArray[5] = SL
         compArray[6] = TP
         compArray[7] = Trade Comment
         compArray[8] = price to execute trade
   */
   
   int switch_action = 0;
   
   if(compArray[0] == "TRADE" && compArray[1] == "OPEN")
      switch_action = 1;
   if(compArray[0] == "RATES")
      switch_action = 2;
   if(compArray[0] == "TRADE" && compArray[1] == "CLOSE")
      switch_action = 3;
   if(compArray[0] == "DATA")
      switch_action = 4;
   
   string ret = "";
   int ticket = -1;
   bool ans = FALSE;
   
   int price_count = 0;
   
   switch(switch_action) 
   {
      case 1: 
         // IMPLEMENT OPEN TRADE LOGIC HERE
         // eurusd_buy_order = "TRADE|OPEN|0|EURUSD|0|50|50|Python-to-MT4|price"
         Print("[ZAWISZA] OPEN TRADE Instruction Received, Trade: "+compArray[1]);
         InformPullClient(pSocket, "[ZAWISZA] OPEN TRADE Instruction Received");
         if(OrdersTotal()>=MaximumOrders){
            Print("[ZAWISZA] Too many open positions");
            ret = "[ZAWISZA] Too many open positions";
            InformPullClient(pSocket, ret); 
            break;
            }
         
         if(compArray[2]==0)
             int ticket=OrderSend(compArray[3],OP_BUY,MaximumLotSize,MarketInfo("EURUSD",MODE_BID),1,compArray[5],compArray[6],compArray[7],MagicNumber,0,clrGreen);
         if(compArray[2]==1)
             int ticket=OrderSend(compArray[3],OP_SELL,MaximumLotSize,MarketInfo("EURUSD",MODE_ASK),1,compArray[5],compArray[6],compArray[7],MagicNumber,0,clrRed);
         
        Print(ticket);
         if(ticket<0)
           {
            Print("[ZAWISZA] OrderSend failed with error #",GetLastError());
            ret = "[ZAWISZA] OrderSend failed with error #" + GetLastError();
           }
         else{
            Print("[ZAWISZA] OrderSend placed successfully");
            ret = "[ZAWISZA] OrderSend placed successfully"+ticket;
         }
         InformPullClient(pSocket, ret); 
         break;
      case 2: 
         ret = "N/A"; 
         if(ArraySize(compArray) > 1) 
            ret = GetBidAsk(compArray[1]); 
            
         InformPullClient(pSocket, ret); 
         break;
      case 3:
         Print("[ZAWISZA] CLOSE TRADE Instruction Received");
         InformPullClient(pSocket, "[ZAWISZA] CLOSE TRADE Instruction Received");
         CloseAllOrders(&pSocket);         
         ret = StringFormat("[ZAWISZA] Trade Closed (Ticket: %d)", ticket);
         InformPullClient(pSocket, ret);
         
         break;
      
      case 4:{
         Print("[ZAWISZA] HISTORICAL DATA Instruction Received");
         InformPullClient(pSocket, "[ZAWISZA] HISTORICAL DATA Instruction Received");
         MqlRates Rates_array[];
         ArraySetAsSeries(Rates_array, true);
         string TimeString="",BarSizeString="",HighString="",LowString="",CloseString="",VolumeString="",HString="";
         //Day(0)        weekday(1)      Hour(2)  minute(3)  BarSize(4)   H(5)    L(6)    C(7)    Volume(8)
         
         // Format: DATA|SYMBOL|TIMEFRAME|START_DATETIME|END_DATETIME
         RefreshRates();
         price_count = CopyRates(compArray[1], StrToInteger(compArray[2]), 
                        StrToInteger(compArray[3]), StrToInteger(compArray[4]), 
                        Rates_array);
         
         if (price_count > 0) {
            for(int i = 0; i < price_count; i++ ) {
               TimeString = TimeString + ";" + TimeToString(Rates_array[i].time,TIME_MINUTES);
               BarSizeString = BarSizeString + ";" + DoubleToStr(Rates_array[i].high - Rates_array[i].low, 5);
               HighString = HighString + ";" + DoubleToStr(Rates_array[i].high, 5);
               LowString = LowString + ";" + DoubleToStr(Rates_array[i].low, 5);
               CloseString = CloseString + ";" + DoubleToStr(Rates_array[i].close, 5);
               VolumeString = VolumeString + ";" + DoubleToStr(Rates_array[i].tick_volume, 1);
               HString = HString + ";" + TimeToString(Rates_array[i].time,TIME_DATE);
            }
            
            ret = "DATA|";
            ret = ret + compArray[1] +"|";
            ret = ret + compArray[2] +"|";
            ret = ret + "TIME"+TimeString +"|";
            ret = ret + "BARSIZE"+BarSizeString +"|";
            ret = ret + "HIGH"+HighString +"|";
            ret = ret + "LOW"+LowString +"|";
            ret = ret + "CLOSE"+CloseString +"|";
            ret = ret + "VOLUME"+VolumeString+"|";
            ret = ret + "Days"+HString+"|";
            ret = ret + "END";
            
            Print("[ZAWISZA] Sending: " + ret);
            InformPullClient(pSocket, ret);
         }
            
         break;
         }
         
      default: 
         break;
   }
}


void CloseAllOrders(Socket &pSocket){
   for(int i=0;i<OrdersTotal();i++)
        {

         if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==false) break;

         if(OrderMagicNumber()!=MagicNumber) continue;
         
         int ticket = OrderTicket();
         //--- check order type 
         if(OrderType()==OP_BUY){

            if(!OrderClose(OrderTicket(),OrderLots(),Bid,3,White)){
               ret = "[ZAWISZA] OrderClose error " + GetLastError();
               Print(ret);
               InformPullClient(pSocket, ret); 
               }
            else{
               ret = "[ZAWISZA] Order Closed, ticket:"+ ticket;
               Print(ret);
               InformPullClient(pSocket, ret);   
               }
            break;
           }
         if(OrderType()==OP_SELL){
            if(!OrderClose(OrderTicket(),OrderLots(),Ask,3,White)){
               ret = "[ZAWISZA] OrderClose error " + GetLastError();
               Print(ret);
               InformPullClient(pSocket, ret); 
               }
            else{
               ret = "[ZAWISZA] Order Closed, ticket:"+ ticket;
               Print(ret);
               InformPullClient(pSocket, ret);   
               }
            break;
           }

            break;
           }
        }

// Parse Zmq Message
void ParseZmqMessage(string& message, string& retArray[]) {
   
   Print("Parsing: " + message);
   
   string sep = "|";
   ushort u_sep = StringGetCharacter(sep,0);
   
   int splits = StringSplit(message, u_sep, retArray);
   
   //for(int i = 0; i < splits; i++) {
   //   Print(i + ") " + retArray[i]);
   //}
}

//+------------------------------------------------------------------+
// Generate string for Bid/Ask by symbol
string GetBidAsk(string symbol) {
   
   double bid = MarketInfo(symbol, MODE_BID);
   double ask = MarketInfo(symbol, MODE_ASK);
   
   return(StringFormat("TICK|%f|%f", bid, ask));
}

// Inform Client
void InformPullClient(Socket& pushSocket, string message) {

   ZmqMsg pushReply(StringFormat("%s", message));
   Print("Is sending working?");
   bool ticket = pushSocket.send(pushReply,true); // NON-BLOCKING
   Print(ticket);
   // pushSocket.send(pushReply,false); // BLOCKING
}

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

   EventSetMillisecondTimer(MILLISECOND_TIMER);     // Set Millisecond Timer to get client socket input
   
   Print("[REP] Binding MT4 Server to Socket on Port " + REP_PORT + "..");   
   Print("[PUSH] Binding MT4 Server to Socket on Port " + PUSH_PORT + "..");
   
   repSocket.bind(StringFormat("%s://%s:%d", ZEROMQ_PROTOCOL, HOSTNAME, REP_PORT));
   pushSocket.bind(StringFormat("%s://%s:%d", ZEROMQ_PROTOCOL, HOSTNAME, PUSH_PORT));
   
   /*
       Maximum amount of time in milliseconds that the thread will try to send messages 
       after its socket has been closed (the default value of -1 means to linger forever):
   */
   
   repSocket.setLinger(1000);  // 1000 milliseconds
   
   /* 
      If we initiate socket.send() without having a corresponding socket draining the queue, 
      we'll eat up memory as the socket just keeps enqueueing messages.
      
      So how many messages do we want ZeroMQ to buffer in RAM before blocking the socket?
   */
   
   repSocket.setSendHighWaterMark(5);     // 5 messages only.
   
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
//---
   Print("[REP] Unbinding MT4 Server from Socket on Port " + REP_PORT + "..");
   repSocket.unbind(StringFormat("%s://%s:%d", ZEROMQ_PROTOCOL, HOSTNAME, REP_PORT));
   
   Print("[PUSH] Unbinding MT4 Server from Socket on Port " + PUSH_PORT + "..");
   pushSocket.unbind(StringFormat("%s://%s:%d", ZEROMQ_PROTOCOL, HOSTNAME, PUSH_PORT));
   
}
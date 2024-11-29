import btfpy

def callback(clientnode,operation,cticn):

  if(operation == btfpy.LE_CONNECT):
    # clientnode has just connected
    print("Connected")
  elif(operation == btfpy.LE_DISCONNECT):
    # clientnode has just disconnected
    print("Disconnected")
    return(btfpy.SERVER_EXIT)
  elif(operation == btfpy.LE_TIMER):
    # The server timer calls here every timerds deci-seconds 
    # clientnode and cticn are invalid
    # This is called by the server not a client  
    # Writing the characteristic sends it as a notification
    btfpy.Write_ctic(btfpy.Localnode(),1,"Hello world",0)
    data = btfpy.Read_ctic(btfpy.Localnode(),2)      # read characteristic index 2
    print(data)
    pass
   
  return(btfpy.SERVER_CONTINUE)


if btfpy.Init_blue("devices.txt") == 0:
  exit(0)

# Set My data (index 1) value  
btfpy.Write_ctic(btfpy.Localnode(),1,"Hello world",0)

btfpy.Le_server(callback,50)   # timerds = 5 seconds
    
btfpy.Close_all()

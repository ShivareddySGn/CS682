const express = require ("express");
const Server = express();
const mongoose = require("mongoose");
const { default: App } = require("./App");
Server.use(express.json());
const mongoUrl ="mongodb+srv://shivareddyh:Lfvps3517k @cluster0.vximcvs.mongodb.net/?retryWrites=true&w=majority"
mongoose
      .connect(mongoUrl,{useNewUrlParser: true}).then(()=>{console.log("Connected to database");}).catch(e=>console.log(e)); 
Server.listen(8000, ()=>{console.log("Server started");});


    
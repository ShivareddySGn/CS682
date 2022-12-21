const { application } = require("express");
const express = require ("express");
const app = express();
const mongoose = require("mongoose");
const { default: App } = require("./App");
const bcrypt = require("bcrypt");
const cors = require("cors");
app.use(cors()); 
app.use(express.json());
const mongoUrl ="mongodb+srv://shivareddyh:Lfvps3517k @cluster0.vximcvs.mongodb.net/?retryWrites=true&w=majority"
 mongoose
      .connect(mongoUrl,{useNewUrlParser: true})
      .then(()=>{console.log("Connected to database");})
      .catch(e=>console.log(e)); 

require("./userDetails");
const User = mongoose.model("userinformation");
app.post("/Signup", async(req, res) => {
    const { email, password } = req.body;
    const encryptedPassword = await bcrypt.hash(password, 10);
    try {
        const oldUser = User.findOne({ email });
        if (oldUser) return res.send("User already exists");
        const user = await User.create({  email, password: encryptedPassword ,}); 
        res.send({ status: "ok" });
    } catch (error) {
        res.send({ status: "error " });
    }
        
    }
); 

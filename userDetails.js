const mongoose = require("mongoose");
const UserDetailsSchema = mongoose.Schema({
    email: {type: String, unique: true}, 
    password: String,
},
 {
    collection : "userinformation"
 }   
    );
mongoose.model("userinformation", UserDetailsSchema);
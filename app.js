const express = require("express"); // Import the express library
const app = express(); // Create an express app
const mongoose = require("mongoose"); // Import the mongoose library for connecting to MongoDB
const bcrypt = require("bcrypt"); // Import the bcrypt library for password hashing
const cors = require("cors"); // Import the cors library for enabling cross-origin resource sharing

// Use the cors middleware to allow requests from any origin
app.use(cors());

// Use the express.json middleware to parse incoming request bodies as JSON
app.use(express.json());

// Connect to the MongoDB database using the mongoUrl
const mongoUrl = "mongodb+srv://shivareddyh:Lfvps3517k@cluster0.vximcvs.mongodb.net/?retryWrites=true&w=majority";
mongoose.connect(mongoUrl, { useNewUrlParser: true })
  .then(() => console.log("Connected to database")) // Log a message if the connection is successful
  .catch(e => console.log(e)); // Log an error if the connection fails

// Require the userDetails module
require("./userDetails");

// Get the userInformation model from the mongoose connection
const User = mongoose.model("userinformation");

// Define a route for handling user sign up
app.post("/Signup", async (req, res) => {
  // Get the email and password from the request body
  const { email, password } = req.body;
  
  // Hash the password using bcrypt
  const encryptedPassword = await bcrypt.hash(password, 10);
  
  try {
    // Check if a user with the same email already exists
    const oldUser = User.findOne({ email });
    if (oldUser) return res.send("User already exists");
    
    // Create a new user with the provided email and encrypted password
    const user = await User.create({ email, password: encryptedPassword });
    
    // Send a response indicating that the user was created successfully
    res.send({ status: "ok" });
  } catch (error) {
    // If there was an error, send a response indicating an error occurred
    res.send({ status: "error" });
  }
});

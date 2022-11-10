
import React from 'react'
import { Grid,Paper, Avatar, TextField, Button, Typography,Link } from '@material-ui/core'
import LockOutlinedIcon from '@material-ui/icons/LockOutlined';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Checkbox from '@material-ui/core/Checkbox';
const pages = ["Adminsignin", "Adminactivities", 'Home'];


const signup=()=>{

    const paperStyle={padding :20,height:'70vh',width:280, margin:"20px auto"}
    const avatarStyle={backgroundColor:'#1bbd7e'}
    const btnstyle={margin:'8px 0'}
    return(
        <Grid>
            <Paper elevation={10} style={paperStyle}>
                <Grid align='center'>
                     <Avatar style={avatarStyle}><LockOutlinedIcon/></Avatar>
                    <h2>Sign up</h2>
                </Grid>
                <TextField label='Email' placeholder='Enter Email' variant="outlined" fullWidth required/>
                <TextField label='Create Password' placeholder='Enter password' type='password' variant="outlined" fullWidth required/>
                <TextField label='confirm password' placeholder='Confirm password' type='password' variant="outlined" fullWidth required/>
                <TextField label='Telephone' placeholder='Enter telephone' type='phone number' variant="outlined" fullWidth required/>

                <FormControlLabel
                    control={
                    <Checkbox
                        name="checkedB"
                        color="primary"
                    />
                    }
                    label="Remember me"
                 />
            <Link href="Adminsignin">   <Button type='submit' color='primary' variant="contained" style={btnstyle} fullWidth >Sign up</Button> </Link> 
                
            </Paper>
        </Grid>
    )
}

export default signup;
import React from 'react'
import {
  Theme,
  createStyles,
  makeStyles,
} from '@material-ui/core/styles'
import Card from '@material-ui/core/Card'
import CardContent from '@material-ui/core/CardContent'
import CardMedia from '@material-ui/core/CardMedia'
import Typography from '@material-ui/core/Typography'
import { Pedestrian } from './Client'

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    root: {
      display: 'flex',
    },
    details: {
      display: 'flex',
      flexDirection: 'column',
    },
    content: {
      flex: '1 0 auto',
    },
    cover: {
      width: 151,
    },
    controls: {
      display: 'flex',
      alignItems: 'center',
      paddingLeft: theme.spacing(1),
      paddingBottom: theme.spacing(1),
    },
    playIcon: {
      height: 38,
      width: 38,
    },
  })
)

export default function PedCard(ped: Pedestrian) {
  const classes = useStyles()
  // const theme = useTheme()
  return (
    <Card className={classes.root}>
      <div className={classes.details}>
        <CardContent className={classes.content}>
          <Typography component="h5" variant="h5">
            {ped.id}
          </Typography>
          <Typography variant="subtitle1" color="textSecondary">
            {ped.confidence}
          </Typography>
        </CardContent>
      </div>
      {ped.image !== null && <CardMedia
        className={classes.cover}
        image={window.URL.createObjectURL(new Blob([ped.image], { type: 'image/jpeg' }))}
        title="Live from space album cover"
      />}
    </Card>
  )
}

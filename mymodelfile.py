
class MyModel:
  def __init__(self):
    #from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    import pandas as pd2
    #self.model = LinearRegression()
    self.model = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    self.pd = pd2
  def fit(self,df):
    ipl_df = df[0]
    ipl_df2 = df[1]
    ipl_df = ipl_df.drop(['non-striker','non_boundary','kind','player_out','fielders_involved'],axis=1)
    ipl_df = ipl_df[ipl_df['overs']<6]
    ipl_df['bowlers'] = ipl_df['bowler'].groupby([ipl_df['ID'],ipl_df['innings']]).transform('nunique')
    ipl_df['wickets'] = ipl_df['isWicketDelivery'].groupby([ipl_df['ID'],ipl_df['innings']]).transform('sum')
    ipl_df3 = self.pd.merge(ipl_df,ipl_df2[['ID','Team1','Team2']],on='ID',how='inner')
    def add(row):
      if(row['BattingTeam'] == row['Team1']):
        return row['Team2']
      else:
        return row['Team1']
    ipl_df3['BowlingTeam'] = ipl_df3.apply(add,axis=1)  
    ipl_df3['pp_score'] = ipl_df3['total_run'].groupby([ipl_df3['ID'],ipl_df3['innings']]).transform('sum')
    ipl_df3 = ipl_df3.drop(['Team1','Team2','overs','ballnumber','batter','bowler','batsman_run','extra_type','extras_run','total_run','isWicketDelivery'],axis=1)
    ipl_df3 = ipl_df3[ipl_df3['BattingTeam'].isin(['Rajasthan Royals', 'Gujarat Titans','Royal Challengers Bangalore', 'Lucknow Super Giants','Sunrisers Hyderabad', 'Punjab Kings', 'Delhi Capitals','Mumbai Indians', 'Chennai Super Kings', 'Kolkata Knight Riders','Kings XI Punjab','Delhi Daredevils'])]
    ipl_df3 = ipl_df3[ipl_df3['BowlingTeam'].isin(['Rajasthan Royals', 'Gujarat Titans','Royal Challengers Bangalore', 'Lucknow Super Giants','Sunrisers Hyderabad', 'Punjab Kings', 'Delhi Capitals','Mumbai Indians', 'Chennai Super Kings', 'Kolkata Knight Riders','Kings XI Punjab','Delhi Daredevils'])]
    def add(row):
      if(row['BattingTeam'] == 'Kings XI Punjab'):
        return 'Punjab Kings'
      else:
        return row['BattingTeam']
    def add2(row):
      if(row['BowlingTeam'] == 'Kings XI Punjab'):
        return 'Punjab Kings'
      else:
        return row['BowlingTeam']
    ipl_df3['BowlingTeam'] = ipl_df3.apply(add2,axis=1)
    ipl_df3['BattingTeam'] = ipl_df3.apply(add,axis=1)
    def add(row):
      if(row['BattingTeam'] == 'Delhi Daredevils'):
        return 'Delhi Capitals'
      else:
        return row['BattingTeam']
    def add2(row):
      if(row['BowlingTeam'] == 'Delhi Daredevils'):
        return 'Delhi Capitals'
      else:
        return row['BowlingTeam']
    ipl_df3['BowlingTeam'] = ipl_df3.apply(add2,axis=1)
    ipl_df3['BattingTeam'] = ipl_df3.apply(add,axis=1)
    ipl_df3 = ipl_df3.drop_duplicates()
    ipl_df3 = self.pd.get_dummies(ipl_df3,columns=['BattingTeam','BowlingTeam']) 
    x = ipl_df3.drop(['ID','pp_score'],axis=1)
    y = ipl_df3['pp_score']
    Q1 = ipl_df3['pp_score'].quantile(0.25)
    Q3 = ipl_df3['pp_score'].quantile(0.75)
    IQR = Q3-Q1
    ipl_df3 = ipl_df3[~((ipl_df3['pp_score']<(Q1-1.5*IQR)) | (ipl_df3['pp_score'] > (Q3 + 1.5 * IQR)))]
    #print(ipl_df3.head(30))
    from sklearn.model_selection import train_test_split
    # Split the data into training and testing sets
    x_train, y_train, x_test, y_test = train_test_split(x, y, test_size = 0.15, random_state = 42)
    
    self.model.fit(x_train,x_test)   
  def predict(self,df):
    w=-2
    #print(df['innings'][0])
    #print(df['bowlers'][0].count(',')+1)
    #print(w+df['batsmen'][0].count(',')+1)
    if(df['batting_team'][0] == 'Rajasthan Royals'):
       arr1 = [0,0,0,0,0,0,0,1,0,0]
    elif(df['batting_team'][0] == 'Mumbai Indians'):
       arr1 = [0,0,0,0,0,1,0,0,0,0]
    elif(df['batting_team'][0] == 'Chennai Super Kings'):
       arr1 = [1,0,0,0,0,0,0,0,0,0]
    elif(df['batting_team'][0] == 'Kolkata Knight Riders'):
       arr1 = [0,0,0,1,0,0,0,0,0,0]
    elif(df['batting_team'][0] == 'Delhi Capitals'):
       arr1 = [0,1,0,0,0,0,0,0,0,0]
    elif(df['batting_team'][0] == 'Punjab Kings'):
       arr1 = [0,0,0,0,0,0,1,0,0,0]
    elif(df['batting_team'][0] == 'Sunrisers Hyderabad'):
       arr1 = [0,0,0,0,0,0,0,0,0,1]
    elif(df['batting_team'][0] == 'Lucknow Super Giants'):
       arr1 = [0,0,0,0,1,0,0,0,0,0]
    elif(df['batting_team'][0] == 'Royal Challengers Bangalore'):
       arr1 = [0,0,0,0,0,0,0,0,1,0]
    elif(df['batting_team'][0] == 'Gujarat Titans'):
       arr1 = [0,0,1,0,0,0,0,0,0,0]

    if(df['bowling_team'][0] == 'Rajasthan Royals'):
       arr2 = [0,0,0,0,0,0,0,1,0,0]
    elif(df['bowling_team'][0] == 'Mumbai Indians'):
       arr2 = [0,0,0,0,0,1,0,0,0,0]
    elif(df['bowling_team'][0] == 'Chennai Super Kings'):
       arr2 = [1,0,0,0,0,0,0,0,0,0]
    elif(df['bowling_team'][0] == 'Kolkata Knight Riders'):
       arr2 = [0,0,0,1,0,0,0,0,0,0]
    elif(df['bowling_team'][0] == 'Delhi Capitals'):
       arr2 = [0,1,0,0,0,0,0,0,0,0]
    elif(df['bowling_team'][0] == 'Punjab Kings'):
       arr2 = [0,0,0,0,0,0,1,0,0,0]
    elif(df['bowling_team'][0] == 'Sunrisers Hyderabad'):
       arr2 = [0,0,0,0,0,0,0,0,0,1]
    elif(df['bowling_team'][0] == 'Lucknow Super Giants'):
       arr2 = [0,0,0,0,1,0,0,0,0,0]
    elif(df['bowling_team'][0] == 'Royal Challengers Bangalore'):
       arr2 = [0,0,0,0,0,0,0,0,1,0]
    elif(df['bowling_team'][0] == 'Gujarat Titans'):
       arr2 = [0,0,1,0,0,0,0,0,0,0]
     #self.model.predict()
    
    a = self.model.predict([[df['innings'][0],df['bowlers'][0].count(',')+1,df['batsmen'][0].count(',')+1] + arr1 + arr2])

    w=-2
    #print(df['innings'][1])
    #print(df['bowlers'][1].count(',')+1)
    #print(w+df['batsmen'][1].count(',')+1)
    if(df['batting_team'][1] == 'Rajasthan Royals'):
       arr3 = [0,0,0,0,0,0,0,1,0,0]
    elif(df['batting_team'][1] == 'Mumbai Indians'):
       arr3 = [0,0,0,0,0,1,0,0,0,0]
    elif(df['batting_team'][1] == 'Chennai Super Kings'):
       arr3 = [1,0,0,0,0,0,0,0,0,0]
    elif(df['batting_team'][1] == 'Kolkata Knight Riders'):
       arr3 = [0,0,0,1,0,0,0,0,0,0]
    elif(df['batting_team'][1] == 'Delhi Capitals'):
       arr3 = [0,1,0,0,0,0,0,0,0,0]
    elif(df['batting_team'][1] == 'Punjab Kings'):
       arr3 = [0,0,0,0,0,0,1,0,0,0]
    elif(df['batting_team'][1] == 'Sunrisers Hyderabad'):
       arr3 = [0,0,0,0,0,0,0,0,0,1]
    elif(df['batting_team'][1] == 'Lucknow Super Giants'):
       arr3 = [0,0,0,0,1,0,0,0,0,0]
    elif(df['batting_team'][1] == 'Royal Challengers Bangalore'):
       arr3 = [0,0,0,0,0,0,0,0,1,0]
    elif(df['batting_team'][1] == 'Gujarat Titans'):
       arr3 = [0,0,1,0,0,0,0,0,0,0]

    if(df['bowling_team'][1] == 'Rajasthan Royals'):
       arr4 = [0,0,0,0,0,0,0,1,0,0]
    elif(df['bowling_team'][1] == 'Mumbai Indians'):
       arr4 = [0,0,0,0,0,1,0,0,0,0]
    elif(df['bowling_team'][1] == 'Chennai Super Kings'):
       arr4 = [1,0,0,0,0,0,0,0,0,0]
    elif(df['bowling_team'][1] == 'Kolkata Knight Riders'):
       arr4 = [0,0,0,1,0,0,0,0,0,0]
    elif(df['bowling_team'][1] == 'Delhi Capitals'):
       arr4 = [0,1,0,0,0,0,0,0,0,0]
    elif(df['bowling_team'][1] == 'Punjab Kings'):
       arr4 = [0,0,0,0,0,0,1,0,0,0]
    elif(df['bowling_team'][1] == 'Sunrisers Hyderabad'):
       arr4 = [0,0,0,0,0,0,0,0,0,1]
    elif(df['bowling_team'][1] == 'Lucknow Super Giants'):
       arr4 = [0,0,0,0,1,0,0,0,0,0]
    elif(df['bowling_team'][1] == 'Royal Challengers Bangalore'):
       arr4 = [0,0,0,0,0,0,0,0,1,0]
    elif(df['bowling_team'][1] == 'Gujarat Titans'):
       arr4 = [0,0,1,0,0,0,0,0,0,0]
     #self.model.predict()
    
    b = self.model.predict([[df['innings'][1],df['bowlers'][1].count(',')+1,df['batsmen'][1].count(',')+1] + arr3 + arr4])
    
    #print(a,b)
   
    return [a,b]

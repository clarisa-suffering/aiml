<form method="POST" action="recommend.php">
  User ID: <input type="text" name="User_ID"><br>
  Calories: <input type="number" name="Calories"><br>
  Protein: <input type="number" name="Protein (g)"><br>
  Carbs: <input type="number" name="Carbohydrates (g)"><br>
  Fat: <input type="number" name="Fat (g)"><br>
  Fiber: <input type="number" name="Fiber (g)"><br>
  Sugar: <input type="number" name="Sugars (g)"><br>
  Sodium: <input type="number" name="Sodium (mg)"><br>
  Cholesterol: <input type="number" name="Cholesterol (mg)"><br>
  Water Intake: <input type="number" name="Water_Intake (ml)"><br>
  Meal Type:
  <select name="Meal_Type">
    <option value="Breakfast">Breakfast</option>
    <option value="Lunch">Lunch</option>
    <option value="Dinner">Dinner</option>
    <option value="Snack">Snack</option>
  </select><br>
  <input type="submit" value="Get Meals">
</form>


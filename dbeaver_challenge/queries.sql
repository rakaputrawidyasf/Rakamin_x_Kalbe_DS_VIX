-- Average customers age, group by "Marital Status"
select "Marital Status", round(avg(age)) as age_avg
from kalbe_customer group by "Marital Status"

-- Average customers age, group by gender
select gender, round(avg(age)) as age_avg
from kalbe_customer group by gender

-- Store with highest sales quantity
select sum(kt.qty) as total_qty, ks.storename
from kalbe_transaction as kt
full outer join kalbe_store as ks
on kt.storeid = ks.storeid
group by ks.storename 
order by total_qty desc
limit 1

-- Product with highest sales total amount
select sum(kt.totalamount) as total_amount_sum, kp."Product Name"
from kalbe_transaction as kt
full outer join kalbe_product as kp
on kt.productid = kp.productid 
group by kp."Product Name" 
order by total_amount_sum desc 
limit 1


with full_table as 
(
	select tc.task_id,proj.id, proj.title, tc.result, tas.data
	from task_completion tc
	inner join task tas on tas.id = tc.task_id
	inner join project proj on proj.id = tas.project_id
	
)

,structured_table as
(
	SELECT task_id,id, title,'http://206.12.96.226:8080/projects/' || id ||'/data?tab=1&task=' || task_id as task_link, --1/data?tab=1&task=42
	(
		SELECT elem->'value'->'choices'->>0
		FROM jsonb_array_elements(result) as elem 
		WHERE elem->>'to_name' = 'IsConflict'
	) AS "IsConflict_choice",
	(
		SELECT elem->'value'->'text'->>0
		FROM jsonb_array_elements(result) as elem 
		WHERE elem->>'to_name' = 'chat'
	) AS "chat_text",
	data->>'tweet_id' AS tweet_id
	FROM full_table
)

--select *  from structured_table where title = 'Annotator 1'

, ann_1 as
(
select * 
from structured_table 
where title = 'Annotator 1'
)

, ann_2 as
(
select * 
from structured_table 
where title = 'Annotator 2'
)

, ann_3 as
(
select * 
from structured_table 
where title = 'Annotator 3'
)

, ann_4 as
(
select * 
from structured_table 
where title = 'Annotator 4'
)

, ann_5 as
(
select * 
from structured_table 
where title = 'Annotator 5'
)

, ann_6 as
(
select * 
from structured_table 
where title = 'Annotator 6'
)

, merged as
(
select a1.tweet_id
, a1.task_link as A_link
, a1."IsConflict_choice" as A_choice
, a1.chat_text as A_chat_text
, a2.task_link as B_link
, a2."IsConflict_choice" as B_choice
, a2.chat_text as B_chat_text
from ann_1 a1 inner join ann_2 a2 on a1.tweet_id = a2.tweet_id

union all 
select a1.tweet_id
, a1.task_link as A_link
, a1."IsConflict_choice" as A_choice
, a1.chat_text as A_chat_text
, a2.task_link as B_link
, a2."IsConflict_choice" as B_choice
, a2.chat_text as B_chat_text
from ann_3 a1 inner join ann_4 a2 on a1.tweet_id = a2.tweet_id

union all 
select a1.tweet_id
, a1.task_link as A_link
, a1."IsConflict_choice" as A_choice
, a1.chat_text as A_chat_text
, a2.task_link as B_link
, a2."IsConflict_choice" as B_choice
, a2.chat_text as B_chat_text
from ann_5 a1 inner join ann_6 a2 on a1.tweet_id = a2.tweet_id
)

select * , case when B_choice <> A_choice then 'Disagreement' else 'Agreement' end as AgreementOrDisagreement 
from merged  
order by AgreementOrDisagreement asc 

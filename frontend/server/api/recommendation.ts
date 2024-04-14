import { RecommendationResult } from "~/models/result";



// const config = useRuntimeConfig()

let environment = process.env.NODE_ENV;
let $endpoint = environment == 'development' ? 'http://localhost:8000/' : 'http://localhost:3333/';

console.log('ENVIRONMENT =', process.env.NODE_ENV)
console.log('$ENDPOINTS =', $endpoint)

export default defineEventHandler(async (event) => {
    if (event.method != 'POST') return sendError(event, Error('Unknown parameters'));

    //console.log('event =', event)

    const body = await readBody<{
        took_and_liked: string,
        took_and_neutral: string,
        took_and_disliked: string,
        curious: string
    }>(event);


    console.log('userData =', body)

    //const {data: responseData} = await useFetch('http://localhost:8000/', {
    const response = await $fetch<RecommendationResult>('http://localhost:8000/recommendations/', {
    
        method: 'post',
        body: { 
            took_and_liked: body.took_and_liked,
            took_and_neutral: body.took_and_neutral,
            took_and_disliked: body.took_and_disliked,
            curious: body.curious
        }
    })

    console.log(response)

    console.log(response.recommendations[0].role)
    console.log(response.recommendations[0].explanation)
    console.log(response.recommendations[0].courses)


    //response.recommendations

    return {
        // ... prediction'dan dönen sonuçları bu return de döndüreceksin
        recommendations: response.recommendations
    } as RecommendationResult;
})
